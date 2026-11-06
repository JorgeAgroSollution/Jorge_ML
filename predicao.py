from __future__ import annotations

import numpy as np
import pandas as pd

import clima
import config as C

SOLO_CAMPOS = (
    "ph_h2o",
    "carbono_organico",
    "ctc",
    "saturacao_bases",
    "aluminio",
    "argila",
    "silte",
    "prof_max_cm",
)


def construir_perfil_local(df: pd.DataFrame, features: list[str], tend) -> pd.DataFrame:
    """Uma linha por município com tudo o que o modelo precisa e o agricultor não informa.

    `tend` é a `alvo.Tendencia` ajustada com todos os dados (a mesma gravada no artefato).
    """
    d = df.copy()
    estaticas = [f for f in features if f != "lag1_rel"]
    g = d.groupby("cod_ibge")
    perfil = g[estaticas].median()
    perfil["lat"], perfil["lon"] = g["lat"].first(), g["lon"].first()
    for extra in ("solo_imputado", "solo_dist_km"):
        if extra in d:
            perfil[extra] = g[extra].first()

    d["_rel"] = d["y_log"] - d.groupby("ano")["y_log"].transform("median")
    d["_anom"] = d["y_log"] - tend.predict(d["ano"])
    ultimo = d.sort_values("ano").groupby("cod_ibge").tail(1).set_index("cod_ibge")
    perfil["lag_ultimo"], perfil["ano_lag"] = ultimo["_rel"], ultimo["ano"]
    perfil["lag_medio"] = d.groupby("cod_ibge")["_rel"].mean()
    perfil["anom_mun"] = d.groupby("cod_ibge")["_anom"].mean()
    perfil["n_anos"] = g["ano"].nunique()
    return perfil.reset_index()


def tendencia_log(artefato: dict, ano: int) -> float:
    """Parte do log-rendimento explicada pela tendência (0 nos artefatos antigos, que não têm tendência)."""
    if artefato.get("target") != "anomalia_y_log":
        return 0.0
    return float(np.polyval(artefato["tendencia_coef"], float(ano)))


def rendimento_kg(artefato: dict, ano: int, saida_modelo: float) -> float:
    """Converte a saída do modelo em kg/ha, respeitando o alvo do artefato (log direto ou anomalia)."""
    return float(np.exp(tendencia_log(artefato, ano) + saida_modelo))


def confiavel(artefato: dict, sem_historico: bool) -> bool:
    """O modelo superou a melhor regra simples disponível nessa situação, por pelo menos SKILL_MINIMO?

    Com o rendimento do ano anterior conhecido, a regra é "tendência + média do município". Sem ele (horizonte
    de 2 anos ou mais) o modelo só tem a média histórica do município, e o teste é o mesmo, sem o lag.
    Em regiões novas (sem histórico) o teste é contra a tendência pura.
    """
    v = artefato.get("validacao", {})
    chave = (
        "skill_temporal_sem_lag_vs_mun+tend"
        if sem_historico
        else "skill_temporal_vs_mun+tend"
    )
    return bool(
        v.get(chave, -1.0) >= C.SKILL_MINIMO
        and v.get("skill_espacial_vs_tendencia", -1.0) >= C.SKILL_MINIMO
    )


def municipio_mais_proximo(
    perfil: pd.DataFrame, lat: float, lon: float
) -> tuple[pd.Series, float]:
    """Município do perfil mais próximo do ponto (haversine) e a distância em km."""
    la, lo = np.deg2rad(perfil["lat"].to_numpy()), np.deg2rad(perfil["lon"].to_numpy())
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    a = (
        np.sin((la - phi) / 2) ** 2
        + np.cos(phi) * np.cos(la) * np.sin((lo - lam) / 2) ** 2
    )
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    i = int(np.argmin(dist))
    return perfil.iloc[i], float(dist[i])


def montar_entrada(
    artefato: dict,
    ano: int,
    lat: float,
    lon: float,
    solo: dict | None = None,
    lag1_rel: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Monta a linha de features do modelo a partir de lat/lon (+ solo opcional). Devolve (entrada, contexto)."""
    perfil = artefato.get("perfil_local")
    if perfil is None:
        raise ValueError(
            "Este artefato não tem 'perfil_local'. Rode `python modelar_culturas.py` para gerá-lo."
        )
    linha, dist = municipio_mais_proximo(perfil, lat, lon)
    if dist > C.DISTANCIA_MAX_KM:
        raise ValueError(
            f"Ponto a {dist:.0f} km do município mais próximo com dados de {artefato.get('cultura', 'cultura')}: "
            "fora da área coberta pelo modelo."
        )
    avisos: list[str] = []
    if dist > C.DISTANCIA_AVISO_KM:
        avisos.append(
            f"O município mais próximo com dados desta cultura fica a {dist:.0f} km; a previsão é regional."
        )

    valores = linha.to_dict()
    informado = {
        k: float(v)
        for k, v in (solo or {}).items()
        if v is not None and k in SOLO_CAMPOS
    }
    valores.update(informado)
    if (
        informado and {"argila", "silte"} <= valores.keys()
    ):  # CAD recalculada com o solo do agricultor
        valores["cad_mm"] = clima.cad_mm(
            valores["argila"], valores["silte"], valores.get("prof_max_cm")
        )
    if informado:
        solo_origem = "informado pelo agricultor"
    elif valores.get("solo_imputado") == 1:
        solo_origem = f"estimado de perfis vizinhos (~{valores.get('solo_dist_km', float('nan')):.0f} km)"
    else:
        solo_origem = "perfil Embrapa do município"

    if lag1_rel is not None:
        lag, lag_origem = float(lag1_rel), "informado"
    elif linha["ano_lag"] == ano - 1:
        lag, lag_origem = float(linha["lag_ultimo"]), "observado (PAM do ano anterior)"
    else:
        lag, lag_origem = float(linha["lag_medio"]), "média histórica do município"
        avisos.append(
            "Sem o rendimento do ano anterior neste município: usada a média histórica local "
            "(a previsão fica menos precisa)."
        )
    valores["lag1_rel"] = lag

    entrada = pd.DataFrame([{f: valores.get(f, np.nan) for f in artefato["features"]}])
    contexto = {
        "cod_ibge": int(linha["cod_ibge"]),
        "distancia_km": dist,
        "solo_origem": solo_origem,
        "lag_origem": lag_origem,
        "anom_mun": float(linha["anom_mun"]),
        "avisos": avisos,
    }
    return entrada, contexto


def prever_simples(
    artefato: dict,
    ano: int,
    lat: float,
    lon: float,
    solo: dict | None = None,
    lag1_rel: float | None = None,
) -> dict:
    """Previsão de rendimento (kg/ha) com faixa P10–P90, usando só os dados informados + o artefato."""
    entrada, ctx = montar_entrada(artefato, ano, lat, lon, solo, lag1_rel)
    saida = float(artefato["modelo"].predict(entrada)[0])
    base = tendencia_log(artefato, ano)
    lo, hi = artefato.get("intervalo_log", (np.nan, np.nan))
    avisos = list(ctx["avisos"])

    if ano - artefato.get("ano_ultimo", ano) > C.HORIZONTE_AVISO_ANOS:
        avisos.append(
            f"Ano além de {artefato['ano_ultimo']}: a tendência de produtividade é apenas extrapolada."
        )
    sem_historico = ctx["lag_origem"].startswith("média")
    ok = confiavel(artefato, sem_historico)
    if not ok:
        avisos.append(
            "Nos testes com dados históricos este modelo não superou a regra simples "
            "(tendência + média do município) nesta situação. Prefira a referência do município."
        )
    return {
        "rendimento_predito_kg_ha": float(np.exp(base + saida)),
        "p10_kg_ha": float(np.exp(base + saida + lo)),
        "p90_kg_ha": float(np.exp(base + saida + hi)),
        "referencia_municipio_kg_ha": float(np.exp(base + ctx["anom_mun"])),
        "municipio_cod_ibge": ctx["cod_ibge"],
        "distancia_km": round(ctx["distancia_km"], 1),
        "solo_origem": ctx["solo_origem"],
        "lag_origem": ctx["lag_origem"],
        "clima_usado": "típico (mediana dos anos de treino)",
        "confiavel": ok,
        "avisos": avisos,
    }
