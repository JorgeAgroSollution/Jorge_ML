"""Clima diário -> ETo (FAO-56), balanço hídrico do solo e features por janela fenológica."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

COLS_DIARIAS = ["temp_media", "temp_max", "temp_min", "precip_mm", "umidade_rel", "radiacao_mj", "vento_ms"]
# Parâmetros adicionais do NASA POWER (via agrobr): pressão (proxy de altitude) e umidade do solo na zona radicular
PARAMETROS_NASA = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M", "PS", "GWETROOT"]


# ---------------------------------------------------------------------------------------------
# Física básica
# ---------------------------------------------------------------------------------------------
def pressao_kpa(altitude_m: float) -> float:
    return 101.3 * ((293.0 - 0.0065 * altitude_m) / 293.0) ** 5.26


def altitude_de_pressao(ps_kpa: float) -> float:
    """Inverso de pressao_kpa. A pressão do NASA POWER é média da célula (~50 km): altitude APROXIMADA."""
    return (293.0 - 293.0 * (ps_kpa / 101.3) ** (1 / 5.26)) / 0.0065


def _e0(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def radiacao_extraterrestre(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))


def eto_fao56(d: pd.DataFrame, lat: float, altitude_m: float) -> np.ndarray:
    """Evapotranspiração de referência (Penman-Monteith FAO-56), mm/dia."""
    tmax, tmin, tmed = d["temp_max"].values, d["temp_min"].values, d["temp_media"].values
    rs = d["radiacao_mj"].values
    u2 = np.clip(d["vento_ms"].values, 0.5, None)  # FAO-56 recomenda piso de 0,5 m/s
    rh = np.clip(d["umidade_rel"].values, 1, 100)
    es = (_e0(tmax) + _e0(tmin)) / 2
    ea = np.minimum(es * rh / 100, es)
    doy = d["data"].dt.dayofyear.values
    ra = radiacao_extraterrestre(lat, doy)
    rso = (0.75 + 2e-5 * altitude_m) * ra
    rs_rso = np.clip(rs / np.maximum(rso, 1e-6), 0.3, 1.0)
    rns = 0.77 * rs
    rnl = 4.903e-9 * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2) * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * rs_rso - 0.35)
    rn = rns - rnl
    gamma = 0.000665 * pressao_kpa(altitude_m)
    delta = 4098 * _e0(tmed) / (tmed + 237.3) ** 2
    eto = (0.408 * delta * rn + gamma * 900 / (tmed + 273) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))
    return np.clip(eto, 0, None)


def balanco_hidrico(precip: np.ndarray, etc: np.ndarray, cad_mm: float, p_depl: float = 0.5) -> np.ndarray:
    """Balanço hídrico diário de um reservatório único. Devolve ETa (mm/dia).

    Ks = 1 enquanto o armazenamento estiver acima de (1-p)*CAD; abaixo disso decresce linearmente.
    Escoamento superficial e drenagem profunda são tratados como excesso (descartado).
    """
    s = 0.5 * cad_mm
    lim = max((1 - p_depl) * cad_mm, 1e-6)
    eta = np.empty(len(precip))
    for i in range(len(precip)):
        s = min(s + precip[i], cad_mm)
        ks = 1.0 if s >= lim else s / lim
        e = min(ks * etc[i], s)
        s -= e
        eta[i] = e
    return eta


def cad_mm(argila_gkg, silte_gkg, prof_cm, prof_max_cm: float = 100.0, padrao: float = 80.0) -> float:
    """Capacidade de água disponível (mm) estimada por função de pedotransferência SIMPLES.

    ATENÇÃO: aproximação grosseira. Substituir por 'agua_dispo'/retenção do perfil (Embrapa) ou por mapas de CAD
    (p. ex. SoilGrids/Embrapa) assim que a unidade desses campos for validada.
    """
    if pd.isna(argila_gkg) or pd.isna(silte_gkg):
        return padrao
    awc_mm_m = np.clip(60 + 0.08 * argila_gkg + 0.12 * silte_gkg, 50, 160)
    z = np.clip(prof_cm if pd.notna(prof_cm) else 100.0, 20, prof_max_cm) / 100.0
    return float(awc_mm_m * z)


# ---------------------------------------------------------------------------------------------
# Preparação do diário e features por janela
# ---------------------------------------------------------------------------------------------
def preparar_diario(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["data"] = pd.to_datetime(d["data"])
    d = d.sort_values("data").drop_duplicates("data").reset_index(drop=True)
    for c in COLS_DIARIAS:
        d[c] = d[c].where(d[c] > -900)  # sentinela -999 do NASA POWER
    d[COLS_DIARIAS] = d[COLS_DIARIAS].interpolate(limit=5)
    d["precip_mm"] = d["precip_mm"].fillna(0).clip(lower=0)
    return d


def altitude_da_celula(d: pd.DataFrame, padrao: float = 500.0) -> float:
    if "ps_kpa" in d and d["ps_kpa"].notna().any():
        return float(altitude_de_pressao(d["ps_kpa"].where(d["ps_kpa"] > 0).mean()))
    return padrao


def _maior_sequencia(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    m = np.concatenate(([0], mask.astype(int), [0]))
    d = np.diff(m)
    ini, fim = np.where(d == 1)[0], np.where(d == -1)[0]
    return int((fim - ini).max())


def features_cultura(d: pd.DataFrame, cultura: str, anos: range, lat: float, altitude_m: float, cad: float) -> pd.DataFrame:
    """Uma linha por ano com estatísticas por janela fenológica da cultura.

    Features por janela (prefixo = nome da janela):
      prec, tmax, tmin, rad            médias/soma básicas
      defic                            soma(ETc - ETa) em mm  -> déficit hídrico
      estresse                         média de (1 - ETa/ETc) -> intensidade do estresse hídrico
      dias_calor, dias_geada           dias com Tmax >= limiar da cultura / Tmin <= 2 °C
      seca_max                         maior sequência de dias com chuva < 1 mm (veranico)
      gd                               graus-dia acima da Tbase da cultura
      umsolo                           umidade da zona radicular (GWETROOT), quando disponível
    + safra_prec, safra_defic: acumulados no intervalo total das janelas.
    """
    cal = C.CALENDARIOS[cultura]
    d = d.copy()
    d["eto"] = eto_fao56(d, lat, altitude_m)
    kc = np.array(cal.kc_mensal)[d["data"].dt.month.values - 1]
    etc = d["eto"].values * kc
    eta = balanco_hidrico(d["precip_mm"].values, etc, cad)

    datas = d["data"].values
    prec, tmax, tmin, rad = d["precip_mm"].values, d["temp_max"].values, d["temp_min"].values, d["radiacao_mj"].values
    tmed2 = (tmax + tmin) / 2
    ums = d["umidade_solo_raiz"].values if "umidade_solo_raiz" in d else np.full(len(d), np.nan)
    ks = np.where(etc > 0.05, eta / np.maximum(etc, 1e-6), 1.0)

    def fatia(ini, fim):
        i0, i1 = np.searchsorted(datas, np.datetime64(ini), "left"), np.searchsorted(datas, np.datetime64(fim), "right")
        n_esp = (fim - ini).days + 1
        if i0 >= len(datas) or datas[0] > np.datetime64(ini) or datas[-1] < np.datetime64(fim) or (i1 - i0) < 0.9 * n_esp:
            return None
        return slice(i0, i1)

    linhas = []
    for ano in anos:
        linha = {"ano": ano}
        ini_total, fim_total = None, None
        for j in cal.janelas:
            ini = pd.Timestamp(ano + j.ano0, j.mes0, 1)
            fim = pd.Timestamp(ano + j.ano1, j.mes1, 1) + pd.offsets.MonthEnd(0)
            ini_total = ini if ini_total is None else min(ini_total, ini)
            fim_total = fim if fim_total is None else max(fim_total, fim)
            s = fatia(ini, fim)
            nomes = ["prec", "tmax", "tmin", "rad", "defic", "estresse", "dias_calor", "dias_geada", "seca_max", "gd", "umsolo"]
            if s is None:
                linha.update({f"{j.nome}_{n}": np.nan for n in nomes})
                continue
            linha[f"{j.nome}_prec"] = prec[s].sum()
            linha[f"{j.nome}_tmax"] = np.nanmean(tmax[s])
            linha[f"{j.nome}_tmin"] = np.nanmean(tmin[s])
            linha[f"{j.nome}_rad"] = np.nanmean(rad[s])
            linha[f"{j.nome}_defic"] = float(np.sum(etc[s] - eta[s]))
            linha[f"{j.nome}_estresse"] = float(np.mean(1 - ks[s]))
            linha[f"{j.nome}_dias_calor"] = int(np.sum(tmax[s] >= cal.t_calor))
            linha[f"{j.nome}_dias_geada"] = int(np.sum(tmin[s] <= 2.0))
            linha[f"{j.nome}_seca_max"] = _maior_sequencia(prec[s] < 1.0)
            linha[f"{j.nome}_gd"] = float(np.sum(np.clip(tmed2[s] - cal.tbase, 0, None)))
            linha[f"{j.nome}_umsolo"] = float(np.nanmean(ums[s])) if np.isfinite(ums[s]).any() else np.nan
        s = fatia(ini_total, fim_total)
        linha["safra_prec"] = prec[s].sum() if s is not None else np.nan
        linha["safra_defic"] = float(np.sum(etc[s] - eta[s])) if s is not None else np.nan
        linhas.append(linha)
    out = pd.DataFrame(linhas)
    # remove colunas sem informação (ex.: umsolo ausente)
    return out.dropna(axis=1, how="all")


# ---------------------------------------------------------------------------------------------
# Coleta (NASA POWER via agrobr) com cache por célula da grade
# ---------------------------------------------------------------------------------------------
def celula(lat: float, lon: float) -> tuple[float, float]:
    """Centro da célula NASA POWER/MERRA-2 (0,5° x 0,625°). Municípios na mesma célula compartilham a requisição."""
    return round(lat / 0.5) * 0.5, round(lon / 0.625) * 0.625


async def coletar_diario(lat_c: float, lon_c: float, ano_ini: int, ano_fim: int) -> pd.DataFrame:
    from agrobr import nasa_power  # import tardio: só necessário na coleta

    cache = C.DIR_CACHE / f"nasa_{lat_c:+.3f}_{lon_c:+.3f}_{ano_ini}_{ano_fim}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = await nasa_power.clima_ponto(
        lat=lat_c, lon=lon_c, inicio=f"{ano_ini}-01-01", fim=f"{ano_fim}-12-31",
        agregacao="diario", parameters=PARAMETROS_NASA,
    )
    df = pd.DataFrame(df)
    df.to_parquet(cache, index=False)
    return df
