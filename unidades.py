"""Nomes de colunas e unidades do dataset exportado.

As funções de modelagem continuam usando nomes internos curtos. O CSV produzido
por ``build_dataset.py`` usa nomes explícitos para que a unidade não dependa de
documentação externa.
"""

from __future__ import annotations

import pandas as pd


_PAM_EXPORTACAO = {
    "rendimento": "rendimento_kg_ha",
    "producao": "producao_t",
    "area_colhida": "area_colhida_ha",
    "area_plantada": "area_plantada_ha",
    "rendimento_calc": "rendimento_calc_kg_ha",
    "valor_producao": "valor_producao_mil_reais",
    "unidade_producao": "unidade_producao_fonte",
    "unidade_rendimento": "unidade_rendimento_fonte",
    "unidade_valor_producao": "unidade_valor_producao_fonte",
}

_SOLO_EXPORTACAO = {
    "carbono_organico": "carbono_organico_g_kg",
    "ctc": "ctc_cmolc_dm3",
    "saturacao_bases": "saturacao_bases_pct",
    "aluminio": "aluminio_cmolc_dm3",
    "fosforo": "fosforo_mg_dm3",
    "argila": "argila_g_kg",
    "silte": "silte_g_kg",
    "prof_max_cm": "prof_max_cm",
}

_CLIMA_EXPORTACAO = {
    "_prec": "_prec_mm",
    "_tmax": "_tmax_c",
    "_tmin": "_tmin_c",
    "_rad": "_rad_mj_m2_dia",
    "_defic": "_defic_mm",
    "_estresse": "_estresse_fracao",
    "_dias_calor": "_dias_calor_d",
    "_dias_geada": "_dias_geada_d",
    "_seca_max": "_seca_max_d",
    "_gd": "_gd_graus_dia",
    "_umsolo": "_umsolo_fracao",
}

_API_PAM_PARA_INTERNO = {
    "rendimento_medio": "rendimento",
    "quantidade_produzida": "producao",
    "rendimento_kg_ha": "rendimento",
    "producao_t": "producao",
    "area_colhida_ha": "area_colhida",
    "area_plantada_ha": "area_plantada",
}


def _renomear_sem_colisao(df: pd.DataFrame, mapa: dict[str, str]) -> pd.DataFrame:
    """Renomeia somente colunas presentes e falha em ambiguidades."""
    presentes = {origem for origem in mapa if origem in df.columns}
    for origem in presentes:
        destino = mapa[origem]
        if destino in df.columns and destino != origem:
            raise ValueError(f"colunas ambíguas: {origem!r} e {destino!r}")
    return df.rename(columns=mapa)


def normalizar_pam(df: pd.DataFrame) -> pd.DataFrame:
    """Converte nomes sem unidade da PAM/agrobr para o esquema interno."""
    return _renomear_sem_colisao(df, _API_PAM_PARA_INTERNO)


def para_exportacao(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica unidades aos nomes do dataset final, sem alterar valores."""
    out = _renomear_sem_colisao(df, {**_PAM_EXPORTACAO, **_SOLO_EXPORTACAO})
    mapa_clima = {}
    for coluna in out.columns:
        for sufixo, novo_sufixo in _CLIMA_EXPORTACAO.items():
            if coluna.endswith(sufixo):
                mapa_clima[coluna] = coluna[: -len(sufixo)] + novo_sufixo
                break
    return _renomear_sem_colisao(out, mapa_clima)


def para_modelagem(df: pd.DataFrame) -> pd.DataFrame:
    """Lê tanto o CSV novo quanto o legado, normalizando para nomes internos."""
    inverso = {v: k for k, v in {**_PAM_EXPORTACAO, **_SOLO_EXPORTACAO}.items()}
    out = _renomear_sem_colisao(df, inverso)
    mapa_clima = {}
    for coluna in out.columns:
        for sufixo, novo_sufixo in _CLIMA_EXPORTACAO.items():
            if coluna.endswith(novo_sufixo):
                mapa_clima[coluna] = coluna[: -len(novo_sufixo)] + sufixo
                break
    return _renomear_sem_colisao(out, mapa_clima)
