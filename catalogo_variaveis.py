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
    "rendimento_kg_ha",
    "producao_t",
    "area_colhida_ha",
    "area_plantada_ha",
    "rendimento_calc_kg_ha",
    "valor_producao_mil_reais",
    "unidade_producao_fonte",
    "unidade_rendimento_fonte",
    "unidade_valor_producao_fonte",
    "razao_colhida_plantada",
    "perda_flag",
    "rend_repetido",
    "y_log",
    "peso",
    "lag1_rel",
]

SOLO = [
    "ph_h2o",
    "carbono_organico_g_kg",
    "ctc_cmolc_dm3",
    "saturacao_bases_pct",
    "aluminio_cmolc_dm3",
    "fosforo_mg_dm3",
    "argila_g_kg",
    "silte_g_kg",
    "agua_dispo",
    "prof_max_cm",
    "cad_mm",
    "n_perfis",
    "solo_imputado",
    "solo_dist_km",
]

# Os indicadores com prefixo de janela aparecem, por exemplo, como
# ``floracao_prec_mm`` e ``enchimento_defic_mm``.
CLIMA_INDICADORES = [
    "prec_mm",
    "tmax_c",
    "tmin_c",
    "rad_mj_m2_dia",
    "defic_mm",
    "estresse_fracao",
    "dias_calor_d",
    "dias_geada_d",
    "seca_max_d",
    "gd_graus_dia",
    "umsolo_fracao",
]

CLIMA_GLOBAIS = [
    "safra_prec_mm",
    "safra_defic_mm",
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
    "rendimento_kg_ha": "Rendimento médio da PAM, em kg/ha.",
    "producao_t": "Quantidade produzida da PAM, em toneladas.",
    "area_colhida_ha": "Área colhida da PAM, em hectares.",
    "area_plantada_ha": "Área plantada da PAM, em hectares.",
    "carbono_organico_g_kg": "Carbono orgânico do solo, em g/kg.",
    "ctc_cmolc_dm3": "Capacidade de troca catiônica, em cmolc/dm3.",
    "saturacao_bases_pct": "Saturação por bases, em %.",
    "aluminio_cmolc_dm3": "Alumínio trocável, em cmolc/dm3.",
    "fosforo_mg_dm3": "Fósforo disponível, em mg/dm3.",
    "argila_g_kg": "Argila, em g/kg.",
    "silte_g_kg": "Silte, em g/kg.",
    "cad_mm": "Capacidade de água disponível estimada, em mm.",
    "solo_imputado": "Indica se os valores de solo foram imputados por vizinhos.",
    "solo_dist_km": "Distância média aos perfis usados na imputação do solo, em km.",
    "n_perfis": "Quantidade de perfis de solo disponíveis no município.",
    "altitude_m": "Altitude aproximada da célula climática, em metros.",
    "safra_prec_mm": "Precipitação acumulada durante todas as janelas da safra, em mm.",
    "safra_defic_mm": "Déficit hídrico acumulado durante todas as janelas, em mm.",
    "valor_producao_mil_reais": "Valor da produção da PAM, em mil reais.",
}
