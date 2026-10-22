"""Tabela de solo por município a partir dos perfis Embrapa — sem misturar horizontes e sem o corte cego de 30% de NaN."""
from __future__ import annotations

import numpy as np
import pandas as pd

import clima
from referencia import normalizar_nome

# Colunas de interesse (nomes do agrobr.embrapa_solos.perfis). Só as que existirem serão usadas.
COLS_QUIMICAS = ["ph_h2o", "carbono_organico", "ctc", "saturacao_bases", "aluminio", "fosforo"]
COLS_TEXTURA = ["argila", "silte"]  # areia = 1000 - argila - silte (colinear => não entra)
COLS_OUTRAS = ["agua_dispo"]  # água disponível medida: unidade a validar antes de usar na CAD


def tabela_solo_municipal(perfis: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Agrega perfis -> município (mediana, robusta a outliers como C.org = 67 g/kg).

    - química e fertilidade: horizontes SUPERFICIAIS (A) — é onde o manejo e as raízes atuam;
    - textura: horizontes A e B (camada 0-100 cm) via mediana;
    - profundidade: máximo por perfil (prof_max_cm), depois mediana no município;
    - mantém fósforo e demais colunas esparsas (o LightGBM/HGB lida com NaN) e registra n_perfis.
    """
    p = perfis.copy()
    p["nome_norm"] = p["municipio"].map(normalizar_nome)
    h = p["horizonte"].astype(str).str.strip() if "horizonte" in p else pd.Series("", index=p.index)
    p["_sup"] = h.str.match(r"^[Aa]", na=False)
    p["_org"] = h.str.match(r"^[OoHh]", na=False)  # horizontes orgânicos distorcem C.org/CTC
    p = p[~p["_org"]]

    chave = "fid" if "fid" in p else None
    prof_perfil = (
        p.groupby(chave)["profundidade"].max() if chave and "profundidade" in p else pd.Series(dtype=float)
    )
    if chave:
        p["_prof_perfil"] = p[chave].map(prof_perfil)

    sup = p[p["_sup"]]
    quim = sup.groupby("nome_norm")[[c for c in COLS_QUIMICAS + COLS_OUTRAS if c in sup]].median()
    p_tex = p[p["profundidade"].fillna(0) <= 100] if "profundidade" in p else p
    tex = p_tex.groupby("nome_norm")[[c for c in COLS_TEXTURA if c in p_tex]].median()
    out = quim.join(tex, how="outer")
    if "_prof_perfil" in p:
        out["prof_max_cm"] = p.groupby("nome_norm")["_prof_perfil"].median()
    out["n_perfis"] = p.groupby("nome_norm")[chave].nunique() if chave else p.groupby("nome_norm").size()
    out["lat_perfil"] = p.groupby("nome_norm")["latitude"].median()
    out["lon_perfil"] = p.groupby("nome_norm")["longitude"].median()

    out = out.reset_index().merge(ref[["cod_ibge", "nome_norm", "lat", "lon"]], on="nome_norm", how="right")
    return out.drop(columns=["nome_norm"])


def imputar_solo_knn(tab: pd.DataFrame, k: int = 3, max_km: float = 80.0) -> pd.DataFrame:
    """Preenche municípios sem perfil com a média dos k vizinhos COM perfil (haversine).

    É uma aproximação: registra `solo_dist_km` (distância média aos vizinhos) e `solo_imputado` (0/1) para que o
    modelo/avaliação possam tratar essas linhas separadamente. Acima de `max_km`, deixa NaN.
    """
    from sklearn.neighbors import BallTree

    cols = [c for c in tab.columns if c in COLS_QUIMICAS + COLS_TEXTURA + COLS_OUTRAS + ["prof_max_cm"]]
    tem = tab["n_perfis"].fillna(0) > 0
    out = tab.copy()
    out["solo_imputado"] = (~tem).astype(int)
    out["solo_dist_km"] = 0.0
    if tem.all() or not tem.any():
        return out
    bt = BallTree(np.deg2rad(tab.loc[tem, ["lat", "lon"]].values), metric="haversine")
    dist, idx = bt.query(np.deg2rad(tab.loc[~tem, ["lat", "lon"]].values), k=min(k, int(tem.sum())))
    dist_km = dist * 6371.0
    base = tab.loc[tem, cols].reset_index(drop=True)
    for pos, (i, di, ii) in enumerate(zip(tab.index[~tem], dist_km, idx)):
        out.loc[i, "solo_dist_km"] = di.mean()
        if di.mean() <= max_km:
            out.loc[i, cols] = base.iloc[ii].mean().values
    return out


def adicionar_cad(tab: pd.DataFrame) -> pd.DataFrame:
    tab = tab.copy()
    nan = pd.Series(np.nan, index=tab.index)
    argila, silte = tab.get("argila", nan), tab.get("silte", nan)
    prof = tab["prof_max_cm"] if "prof_max_cm" in tab else nan
    tab["cad_mm"] = [clima.cad_mm(a, s, p) for a, s, p in zip(argila, silte, prof)]
    return tab
