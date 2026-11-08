"""Exporta as previsões pré-calculadas que o Jorge carrega no banco (assim ele não precisa de serviço Python no ar).

Para cada cultura, cada município do `perfil_local` e os anos `ano_ultimo + 1` até `ano_ultimo + N`, roda o
mesmo cálculo de `POST /predicoes/simples` (predicao.montar_entrada + predicao.resultado_previsao) no ponto
do município. Uso:

    uv run python exportar_previsoes.py
    uv run python exportar_previsoes.py --anos 2 --saida resultados/previsoes_jorge.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

import api
import config as C
import predicao

FORMATO = 1  # muda quando os campos do JSON mudarem (entra na `versao`)
SAIDA_PADRAO = C.DIR_RESULTADOS / "previsoes_jorge.json"
MUNICIPIOS_PADRAO = C.DIR_RAW / "municipios_ref.csv"


def nomes_municipios(caminho: Path) -> dict[int, str]:
    """Código IBGE -> nome do município (data/raw/municipios_ref.csv)."""
    ref = pd.read_csv(caminho, usecols=["cod_ibge", "municipio"])
    return {int(c): str(n) for c, n in zip(ref["cod_ibge"], ref["municipio"])}


def linhas_cultura(cultura: str, artefato: dict, anos: list[int], nomes: dict[int, str]) -> list[dict]:
    """Previsão de cada município do perfil local para cada ano, com uma chamada ao modelo por ano."""
    perfil = artefato["perfil_local"]
    linhas = []
    for ano in anos:
        entradas, contextos = [], []
        for lat, lon in zip(perfil["lat"], perfil["lon"]):
            entrada, ctx = predicao.montar_entrada(artefato, ano, float(lat), float(lon))
            entradas.append(entrada)
            contextos.append(ctx)
        saidas = artefato["modelo"].predict(pd.concat(entradas, ignore_index=True))
        vistos: set[int] = set()
        for (lat, lon), ctx, saida in zip(zip(perfil["lat"], perfil["lon"]), contextos, saidas):
            if ctx["cod_ibge"] in vistos:  # dois municípios com o mesmo centroide: fica o primeiro
                continue
            vistos.add(ctx["cod_ibge"])
            r = predicao.resultado_previsao(artefato, ano, ctx, float(saida))
            codigos = r["avisos_codigos"]
            linha = {
                "cultura": cultura,
                "cod_ibge": ctx["cod_ibge"],
                "municipio": nomes.get(ctx["cod_ibge"], str(ctx["cod_ibge"])),
                "lat": round(float(lat), 4),
                "lon": round(float(lon), 4),
                "ano": ano,
                "kg_ha": round(r["rendimento_predito_kg_ha"]),
                "p10_kg_ha": round(r["p10_kg_ha"]),
                "p90_kg_ha": round(r["p90_kg_ha"]),
                "referencia_kg_ha": round(r["referencia_municipio_kg_ha"]),
                "confiavel": r["confiavel"],
                "lag_origem": "media" if predicao.AVISO_SEM_LAG in codigos else "observado",
            }
            # Só vai quando é verdadeiro: com os 3 anos padrão nunca é, e o campo sempre falso passava de 3 MB.
            if predicao.AVISO_EXTRAPOLACAO in codigos:
                linha["extrapolado"] = True
            linhas.append(linha)
    return linhas


def versao_exportacao(modelos: dict[str, dict], n_anos: int) -> str:
    """sha256 curto do formato + versões dos artefatos: só muda quando algum modelo ou o formato muda."""
    chave = json.dumps(
        {"formato": FORMATO, "anos": n_anos, "modelos": {c: m["versao"] for c, m in sorted(modelos.items())}},
        sort_keys=True,
    )
    return hashlib.sha256(chave.encode()).hexdigest()[:12]


def exportar(pasta_modelos: Path, municipios_csv: Path, n_anos: int = C.HORIZONTE_AVISO_ANOS) -> dict:
    """Monta o documento exportado: versão, data, metadados dos modelos e a lista de linhas."""
    nomes = nomes_municipios(municipios_csv)
    modelos: dict[str, dict] = {}
    linhas: list[dict] = []
    for cultura in C.CULTURAS:
        caminho = pasta_modelos / f"modelo_{cultura}.joblib"
        if not caminho.exists():
            continue
        artefato = joblib.load(caminho)
        ano_ultimo = int(artefato.get("ano_ultimo", C.ANO_FINAL))
        anos = list(range(ano_ultimo + 1, ano_ultimo + 1 + n_anos))
        modelos[cultura] = {
            "versao": api.versao_artefato(caminho),
            "ano_ultimo": ano_ultimo,
            "modelo": artefato.get("modelo_nome", type(artefato["modelo"]).__name__),
        }
        linhas.extend(linhas_cultura(cultura, artefato, anos, nomes))
    return {
        "versao": versao_exportacao(modelos, n_anos),
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "formato": FORMATO,
        "unidade": "kg/ha",
        "modelos": modelos,
        "linhas": linhas,
    }


def gravar(documento: dict, saida: Path) -> int:
    """Grava o JSON compacto (sem espaços) e devolve o tamanho em bytes."""
    saida.parent.mkdir(parents=True, exist_ok=True)
    texto = json.dumps(documento, ensure_ascii=False, separators=(",", ":"))
    saida.write_text(texto, encoding="utf-8")
    return len(texto.encode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", type=Path, default=C.DIR_RESULTADOS / "modelos_cultura")
    ap.add_argument("--municipios", type=Path, default=MUNICIPIOS_PADRAO)
    ap.add_argument("--anos", type=int, default=C.HORIZONTE_AVISO_ANOS, help="anos após o último dado")
    ap.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    args = ap.parse_args()
    documento = exportar(args.modelos, args.municipios, args.anos)
    tamanho = gravar(documento, args.saida)
    print(
        f"{len(documento['linhas'])} linhas de {len(documento['modelos'])} culturas, "
        f"versão {documento['versao']}, {tamanho / 1e6:.2f} MB em {args.saida}"
    )


if __name__ == "__main__":
    main()
