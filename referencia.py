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

    Fonte: tabela municipal do IBGE e centróides distribuídos pelo agrobr.
    Cacheado em data/raw/municipios_ref.csv.
    """
    arq = C.DIR_RAW / "municipios_ref.csv"
    if arq.exists() and not forcar:
        return pd.read_csv(arq, dtype={"cod_ibge": "int64"})
    from agrobr import ibge
    from agrobr.normalize.municipalities import _load_municipios

    malha = await ibge.malha_municipal(uf=C.UF)
    col_nome = next(c for c in ("nm_mun", "nome", "municipio", "nm_municipio") if c in malha.columns)
    coords = pd.DataFrame(_load_municipios(), columns=["cod_ibge", "_municipio", "_uf", "lat", "lon"])
    coords["cod_ibge"] = coords["cod_ibge"].astype("int64")
    coords = coords[coords["_uf"].eq(C.UF)]
    gdf = malha.rename(columns={"cod_municipio": "cod_ibge"}).merge(
        coords[["cod_ibge", "lat", "lon"]], on="cod_ibge", how="left"
    )
    if gdf[["lat", "lon"]].isna().any().any():
        faltantes = gdf.loc[gdf[["lat", "lon"]].isna().any(axis=1), "cod_ibge"].tolist()
        raise RuntimeError(f"agrobr não forneceu coordenadas para municípios: {faltantes[:10]}")
    ref = pd.DataFrame(
        {
            "cod_ibge": gdf["cod_ibge"].astype("int64"),
            "municipio": gdf[col_nome],
            "lat": gdf["lat"],
            "lon": gdf["lon"],
        }
    )
    ref["nome_norm"] = ref["municipio"].map(normalizar_nome)
    assert ref["nome_norm"].is_unique, "nomes normalizados duplicados em SP: usar apenas o código IBGE"
    ref.to_csv(arq, index=False)
    return ref
