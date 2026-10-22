"""Tabela de referência de municípios (código IBGE, nome normalizado, centroide) e utilitários de nome."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

import config as C


def normalizar_nome(valor) -> object:
    """minúsculas, sem acento, sem ' - SP', espaços colapsados. Usar SÓ para ligar tabelas sem código IBGE."""
    if pd.isna(valor):
        return pd.NA
    v = str(valor)
    try:  # corrige mojibake (utf-8 lido como cp1252)
        v = v.encode("cp1252").decode("utf-8")
    except (UnicodeError, ValueError):
        pass
    v = re.sub(r"\s*-\s*sp\s*$", "", v.strip().casefold())
    v = "".join(c for c in unicodedata.normalize("NFKD", v) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", v)


async def construir_referencia(forcar: bool = False) -> pd.DataFrame:
    """cod_ibge, municipio, nome_norm, lat, lon (ponto representativo dentro do polígono).

    Fonte: malha municipal do IBGE via agrobr (requer geopandas). Cacheado em data/raw/municipios_ref.csv.
    """
    arq = C.DIR_RAW / "municipios_ref.csv"
    if arq.exists() and not forcar:
        return pd.read_csv(arq, dtype={"cod_ibge": "int64"})
    from agrobr import ibge

    malha = await ibge.malha_municipal(uf=C.UF)
    gdf = malha.to_crs(5880)  # SIRGAS 2000 / Polyconic: projeção métrica p/ o Brasil
    pt = gdf.geometry.representative_point().to_crs(4326)
    col_nome = next(c for c in ("nm_mun", "nome", "municipio", "nm_municipio") if c in gdf.columns)
    ref = pd.DataFrame(
        {"cod_ibge": gdf["cod_municipio"].astype("int64"), "municipio": gdf[col_nome], "lat": pt.y.values, "lon": pt.x.values}
    )
    ref["nome_norm"] = ref["municipio"].map(normalizar_nome)
    assert ref["nome_norm"].is_unique, "nomes normalizados duplicados em SP: usar apenas o código IBGE"
    ref.to_csv(arq, index=False)
    return ref
