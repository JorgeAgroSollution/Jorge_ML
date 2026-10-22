"""Limpeza da PAM e construção do alvo (anomalia de produtividade em relação à tendência tecnológica)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _runs_repetidos(df: pd.DataFrame) -> pd.Series:
    """Tamanho da sequência de anos consecutivos com rendimento idêntico (por município-cultura)."""
    df = df.sort_values(["produto", "localidade", "ano"])
    chave = df["produto"].astype(str) + "|" + df["localidade"].astype(str)
    mudou = (df["rendimento"] != df.groupby(chave)["rendimento"].shift()) | (chave != chave.shift())
    run_id = mudou.cumsum()
    return df.groupby(run_id)["rendimento"].transform("size").reindex(df.index)


def limpar_pam(df: pd.DataFrame, area_min: float = C.AREA_COLHIDA_MIN_HA, descartar_repetidos: bool = False):
    """Aplica as regras de qualidade ao alvo e devolve (df_limpo, relatorio).

    O relatório lista quantas linhas cada regra removeu (por cultura), para auditoria.

    Valores repetidos por >= RUN_REPETIDO_MAX anos (estimativas, não medições) NÃO são descartados por padrão
    (removeriam 20-40% da base, p. ex. cana); recebem peso reduzido (x0,3). Use descartar_repetidos=True para excluir.
    """
    df = df.copy()
    # remove colunas duplicadas herdadas do merge antigo
    df = df.drop(columns=[c for c in C.COLS_REDUNDANTES if c in df.columns and c.startswith(("latitude_", "longitude_"))])
    # dropna só nas colunas essenciais (o dropna() global antigo descartava linhas por motivos alheios ao alvo)
    df = df.dropna(subset=["rendimento", "area_colhida", "produto", "localidade", "ano"])
    df = df[(df["rendimento"] > 0) & (df["area_colhida"] > 0)]

    df["razao_colhida_plantada"] = df["area_colhida"] / df["area_plantada"].where(df["area_plantada"] > 0)
    df["perda_flag"] = (df["razao_colhida_plantada"] < 0.9).astype(int)
    df["rendimento_calc"] = df["producao"] * 1000 / df["area_colhida"]

    df["_run"] = _runs_repetidos(df)
    df["rend_repetido"] = (df["_run"] >= 2).astype(int)

    # mediana do município-cultura (para detectar saltos absurdos)
    med = df.groupby(["produto", "localidade"])["rendimento"].transform("median")
    df["_razao_mun"] = df["rendimento"] / med

    regras = {
        "quebra_de_serie": df.apply(lambda r: r["ano"] < C.PRIMEIRO_ANO_VALIDO.get(r["produto"], 0), axis=1),
        f"area_colhida<{area_min}ha": df["area_colhida"] < area_min,
        "fora_dos_limites_agronomicos": df.apply(
            lambda r: not (C.LIMITES_RENDIMENTO.get(r["produto"], (0, np.inf))[0] <= r["rendimento"] <= C.LIMITES_RENDIMENTO.get(r["produto"], (0, np.inf))[1]),
            axis=1,
        ),
        "salto_vs_mediana_municipal": (df["_razao_mun"] < C.RAZAO_MUN_MIN) | (df["_razao_mun"] > C.RAZAO_MUN_MAX),
    }
    if descartar_repetidos:
        regras[f"estimado(run>={C.RUN_REPETIDO_MAX}anos_iguais)"] = df["_run"] >= C.RUN_REPETIDO_MAX

    ficou = pd.Series(True, index=df.index)
    linhas = []
    for nome, mask in regras.items():
        novos = mask & ficou
        resumo = df.loc[novos].groupby("produto").size().rename(nome)
        linhas.append(resumo)
        ficou &= ~mask
    relatorio = pd.concat(linhas, axis=1).fillna(0).astype(int)
    relatorio["linhas_iniciais"] = df.groupby("produto").size()
    relatorio["linhas_finais"] = df[ficou].groupby("produto").size()
    relatorio = relatorio.fillna(0).astype(int)

    limpo = df[ficou].drop(columns=["_razao_mun"]).reset_index(drop=True)
    limpo["y_log"] = np.log(limpo["rendimento"])
    limpo["peso"] = np.sqrt(limpo["area_colhida"])  # municípios grandes têm estimativa mais confiável
    if not descartar_repetidos:
        limpo["peso"] = limpo["peso"] * np.where(limpo["_run"] >= C.RUN_REPETIDO_MAX, 0.3, 1.0)
    limpo = limpo.drop(columns=["_run"])
    return limpo, relatorio


class Tendencia:
    """Tendência tecnológica por cultura: reta ajustada às medianas anuais de log-rendimento.

    Deve ser AJUSTADA SOMENTE NO TREINO de cada fold (senão há vazamento temporal).
    """

    def __init__(self):
        self.coef = (0.0, 0.0)

    def fit(self, ano: pd.Series, y_log: pd.Series, grau: int = 1) -> "Tendencia":
        med = pd.DataFrame({"ano": ano.values, "y": y_log.values}).groupby("ano")["y"].median()
        if len(med) >= 3:
            self.coef = tuple(np.polyfit(med.index.values.astype(float), med.values, grau))
        else:
            self.coef = (0.0,) * grau + (float(med.mean()),)
        return self

    def predict(self, ano) -> np.ndarray:
        return np.polyval(self.coef, np.asarray(ano, dtype=float))
