"""Catálogo das variáveis produzidas pelo dataset agrícola.

O módulo centraliza a separação das colunas por origem para exploração,
relatórios e preparação de dados. As colunas climáticas são geradas por
janela fenológica e, por isso, são identificadas pelos sufixos documentados
em ``CLIMA_INDICADORES``.

Exemplo:

    import pandas as pd
    from catalogo_variaveis import selecionar_grupos

    df = pd.read_csv("data/processed/dataset_completo.csv")
    grupos = selecionar_grupos(df.columns)
    clima = df[grupos["clima"]]
    solo = df[grupos["solo"]]
"""

from __future__ import annotations

from collections.abc import Iterable


CHAVES = [
    "cod_ibge",
    "produto",
    "ano",
    "localidade",
]

LOCALIZACAO = [
    "lat",
    "lon",
    "latitude",
    "longitude",
]

PAM = [
    "rendimento",
    "producao",
    "area_colhida",
    "area_plantada",
    "rendimento_calc",
    "razao_colhida_plantada",
    "perda_flag",
    "rend_repetido",
    "y_log",
    "peso",
    "lag1_rel",
]

SOLO = [
    "ph_h2o",
    "carbono_organico",
    "ctc",
    "saturacao_bases",
    "aluminio",
    "fosforo",
    "argila",
    "silte",
    "agua_dispo",
    "prof_max_cm",
    "cad_mm",
    "n_perfis",
    "solo_imputado",
    "solo_dist_km",
]

# Os indicadores com prefixo de janela aparecem, por exemplo, como
# ``floracao_prec`` e ``enchimento_defic``.
CLIMA_INDICADORES = [
    "prec",
    "tmax",
    "tmin",
    "rad",
    "defic",
    "estresse",
    "dias_calor",
    "dias_geada",
    "seca_max",
    "gd",
    "umsolo",
]

CLIMA_GLOBAIS = [
    "safra_prec",
    "safra_defic",
    "altitude_m",
]


def _eh_coluna_climatica(coluna: str) -> bool:
    """Retorna se a coluna pertence às features climáticas."""
    if coluna in CLIMA_GLOBAIS:
        return True
    return any(coluna.endswith(f"_{indicador}") for indicador in CLIMA_INDICADORES)


def selecionar_grupos(colunas: Iterable[str]) -> dict[str, list[str]]:
    """Separa as colunas existentes no dataset por origem.

    A ordem de cada grupo segue a ordem declarada neste módulo, e não a ordem
    recebida em ``colunas``. Colunas não catalogadas ficam em ``outras`` para
    que novas variáveis não sejam ignoradas silenciosamente.
    """
    colunas = list(colunas)
    existentes = set(colunas)
    grupos = {
        "chaves": [coluna for coluna in CHAVES if coluna in existentes],
        "localizacao": [coluna for coluna in LOCALIZACAO if coluna in existentes],
        "pam": [coluna for coluna in PAM if coluna in existentes],
        "clima": [
            coluna
            for coluna in colunas
            if _eh_coluna_climatica(coluna)
        ],
        "solo": [coluna for coluna in SOLO if coluna in existentes],
    }
    catalogadas = set().union(*grupos.values())
    grupos["outras"] = [coluna for coluna in colunas if coluna not in catalogadas]
    return grupos


DESCRICOES = {
    "cad_mm": "Capacidade de água disponível estimada no solo, em mm.",
    "solo_imputado": "Indica se os valores de solo foram imputados por vizinhos.",
    "solo_dist_km": "Distância média aos perfis usados na imputação do solo, em km.",
    "n_perfis": "Quantidade de perfis de solo disponíveis no município.",
    "altitude_m": "Altitude aproximada da célula climática, em metros.",
    "safra_prec": "Precipitação acumulada durante todas as janelas da safra, em mm.",
    "safra_defic": "Déficit hídrico acumulado durante todas as janelas, em mm.",
}
