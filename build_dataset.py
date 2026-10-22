"""Pipeline de coleta e montagem do dataset (uma tabela por cultura), reprodutível de ponta a ponta.

Uso:
    python build_dataset.py --pam-csv data/raw/df_market_pam.csv           # reaproveita a PAM já baixada
    python build_dataset.py --coletar-pam                                  # baixa a PAM via agrobr
    python build_dataset.py --pam-csv ... --sem-knn                        # só municípios COM perfil de solo

Saídas (data/processed): dataset_<cultura>.csv, dataset_completo.csv e relatorio_limpeza.csv.

Diferenças em relação ao eda.py original:
  * junta tudo por CÓDIGO IBGE (nome normalizado só como ponte quando a PAM não traz o código);
  * clima DIÁRIO por célula da grade (cache) -> ETo, balanço hídrico e features por janela fenológica da cultura;
  * solo por município sem misturar horizontes; municípios sem perfil recebem solo dos vizinhos (marcado);
  * dropna só no que é essencial; limpeza explícita e auditável do alvo;
  * não usa produção/área como feature; inclui rendimento defasado RELATIVO (lag1_rel).
"""
from __future__ import annotations

import argparse
import asyncio

import numpy as np
import pandas as pd

import alvo
import clima
import config as C
import referencia
import solo


def carregar_pam(caminho) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    df = df.drop(columns=["fonte"], errors="ignore")
    obrig = {"localidade", "ano", "produto", "rendimento", "area_colhida", "area_plantada", "producao"}
    faltam = obrig - set(df.columns)
    assert not faltam, f"PAM sem colunas obrigatórias: {faltam}"
    return df


async def coletar_pam() -> pd.DataFrame:
    """Baixa a PAM municipal de SP. Ajuste aqui se a versão do agrobr devolver outro formato (p. ex. longo)."""
    from agrobr import ibge

    quadros = []
    for cultura in C.CULTURAS:
        df = pd.DataFrame(
            await ibge.pam(cultura, ano=list(range(C.ANO_INICIAL, C.ANO_FINAL + 1)), uf=C.UF, nivel="municipio")
        )
        df["produto"] = cultura
        quadros.append(df)
    pam = pd.concat(quadros, ignore_index=True)
    pam = pam.rename(columns={"quantidade_produzida": "producao", "rendimento_medio": "rendimento"})
    return pam


def ligar_codigo_ibge(pam: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    if "localidade_cod" in pam.columns:
        pam = pam.assign(cod_ibge=pd.to_numeric(pam["localidade_cod"], errors="coerce").astype("Int64"))
    else:
        pam = pam.assign(nome_norm=pam["localidade"].map(referencia.normalizar_nome))
        pam = pam.merge(ref[["cod_ibge", "nome_norm"]], on="nome_norm", how="left").drop(columns="nome_norm")
    sem = pam["cod_ibge"].isna()
    if sem.any():
        print(f"[aviso] {sem.sum()} linhas da PAM sem código IBGE (descartadas): {pam.loc[sem, 'localidade'].unique()[:10]}")
    return pam[~sem].astype({"cod_ibge": "int64"})


async def obter_solo(ref: pd.DataFrame, usar_knn: bool) -> pd.DataFrame:
    arq = C.DIR_RAW / "perfis_sp.parquet"
    if arq.exists():
        perfis = pd.read_parquet(arq)
    else:
        from agrobr import embrapa_solos

        perfis = pd.DataFrame(await embrapa_solos.perfis(uf=C.UF))
        perfis.to_parquet(arq, index=False)
    tab = solo.tabela_solo_municipal(perfis, ref)
    if usar_knn:
        tab = solo.imputar_solo_knn(tab)
    else:
        tab["solo_imputado"], tab["solo_dist_km"] = 0, 0.0
        tab = tab[tab["n_perfis"].fillna(0) > 0]
    return solo.adicionar_cad(tab)


async def obter_clima_features(pam: pd.DataFrame, ref: pd.DataFrame, tab_solo: pd.DataFrame) -> pd.DataFrame:
    """Features por (cod_ibge, cultura, ano). Uma requisição por célula da grade NASA POWER."""
    pares = pam.groupby("cod_ibge")["produto"].unique()
    info = ref.set_index("cod_ibge").join(tab_solo.set_index("cod_ibge")[["cad_mm"]], how="left")
    info = info.loc[info.index.intersection(pares.index)]
    info["cel"] = [clima.celula(la, lo) for la, lo in zip(info["lat"], info["lon"])]
    saidas = []
    cel_ids = info["cel"].unique()
    for n, cel in enumerate(cel_ids, 1):
        print(f"[clima] célula {n}/{len(cel_ids)} {cel}", flush=True)
        diario = clima.preparar_diario(await clima.coletar_diario(*cel, C.ANO_INICIAL - 1, C.ANO_FINAL))
        alt = clima.altitude_da_celula(diario)
        for cod, linha in info[info["cel"] == cel].iterrows():
            cad = linha["cad_mm"] if pd.notna(linha["cad_mm"]) else 80.0
            for cultura in pares[cod]:
                if cultura not in C.CALENDARIOS:
                    continue
                f = clima.features_cultura(diario, cultura, range(C.ANO_INICIAL, C.ANO_FINAL + 1), linha["lat"], alt, cad)
                f.insert(0, "cod_ibge", cod)
                f.insert(1, "produto", cultura)
                f["altitude_m"] = alt
                saidas.append(f)
    return pd.concat(saidas, ignore_index=True)


def adicionar_lag(df: pd.DataFrame) -> pd.DataFrame:
    """lag1_rel = log-rendimento do ano anterior do município MENOS a mediana (todos os municípios) daquele ano.

    É relativo ao ano para não carregar a tendência tecnológica; só usa informação já conhecida no plantio.
    """
    med_ano = df.groupby(["produto", "ano"])["y_log"].median().rename("_med")
    d = df.join(med_ano, on=["produto", "ano"])
    d["_rel"] = d["y_log"] - d["_med"]
    prev = d[["cod_ibge", "produto", "ano", "_rel"]].assign(ano=lambda x: x["ano"] + 1).rename(columns={"_rel": "lag1_rel"})
    return df.merge(prev, on=["cod_ibge", "produto", "ano"], how="left")


async def main(args):
    C.DIR_PROC.mkdir(parents=True, exist_ok=True)
    ref = await referencia.construir_referencia()
    pam = carregar_pam(args.pam_csv) if args.pam_csv else await coletar_pam()
    pam = ligar_codigo_ibge(pam, ref)
    if args.culturas:
        pam = pam[pam["produto"].isin(args.culturas)]
    pam, relatorio = alvo.limpar_pam(pam)
    relatorio.to_csv(C.DIR_PROC / "relatorio_limpeza.csv")
    print(relatorio.to_string())

    tab_solo = await obter_solo(ref, usar_knn=not args.sem_knn)
    pam = pam[pam["cod_ibge"].isin(tab_solo["cod_ibge"])]
    feats = await obter_clima_features(pam, ref, tab_solo)

    solo_cols = [c for c in tab_solo.columns if c not in {"lat_perfil", "lon_perfil"}]
    df = (
        pam.merge(feats, on=["cod_ibge", "produto", "ano"], how="inner")
        .merge(tab_solo[solo_cols], on="cod_ibge", how="left")
        .drop(columns=[c for c in ("localidade_cod",) if c in pam])
    )
    df = adicionar_lag(df)
    assert not df.duplicated(["cod_ibge", "produto", "ano"]).any()
    df.to_csv(C.DIR_PROC / "dataset_completo.csv", index=False)
    for cultura, g in df.groupby("produto"):
        g.to_csv(C.DIR_PROC / f"dataset_{cultura}.csv", index=False)
    print(f"dataset: {df.shape}, culturas: {df['produto'].nunique()}, municípios: {df['cod_ibge'].nunique()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pam-csv")
    g.add_argument("--coletar-pam", action="store_true")
    ap.add_argument("--sem-knn", action="store_true", help="não imputa solo de vizinhos (só municípios com perfil)")
    ap.add_argument("--culturas", nargs="*")
    asyncio.run(main(ap.parse_args()))
