"""Testes da predição com o mínimo de dados (cultura, ano, latitude, longitude). Sem rede.

    uv run pytest -q test_predicao.py
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
import modelar_culturas
import predicao


def _dataset_sintetico(n_mun=40, anos=range(2000, 2026), seed=0) -> pd.DataFrame:
    """Rendimento = nível do município x tendência de +2%/ano x efeito da argila x ruído."""
    rng = np.random.default_rng(seed)
    mun = pd.DataFrame(
        {
            "cod_ibge": 3500000 + np.arange(n_mun),
            "lat": rng.uniform(-24, -20, n_mun),
            "lon": rng.uniform(-51, -46, n_mun),
            "argila": rng.uniform(100, 600, n_mun),
            "silte": rng.uniform(30, 150, n_mun),
            "nivel": rng.normal(0, 0.15, n_mun),
        }
    )
    linhas = []
    for _, m in mun.iterrows():
        lag_rel_prev = 0.0
        for ano in anos:
            clima_janela = rng.normal(0, 1)
            y_log = (
                np.log(3000)
                + 0.02 * (ano - 2000)
                + m["nivel"]
                + 0.0004 * (m["argila"] - 350)
                + 0.03 * clima_janela
                + rng.normal(0, 0.05)
            )
            linhas.append(
                {
                    "cod_ibge": int(m["cod_ibge"]),
                    "localidade": f"mun{int(m['cod_ibge'])}",
                    "produto": "milho",
                    "ano": ano,
                    "lat": m["lat"],
                    "lon": m["lon"],
                    "argila": m["argila"],
                    "silte": m["silte"],
                    "prof_max_cm": 100.0,
                    "cad_mm": 50.0 + 0.05 * m["argila"],
                    "altitude_m": 400.0 + 200.0 * m["nivel"],
                    "floracao_defic": clima_janela,
                    "y_log": y_log,
                    "rendimento": float(np.exp(y_log)),
                    "peso": 10.0,
                    "solo_imputado": 0,
                    "solo_dist_km": 0.0,
                }
            )
    df = pd.DataFrame(linhas)
    rel = df["y_log"] - df.groupby("ano")["y_log"].transform("median")
    df["_rel"] = rel
    prev = df[["cod_ibge", "ano", "_rel"]].assign(ano=lambda x: x["ano"] + 1)
    df = df.merge(prev.rename(columns={"_rel": "lag1_rel"}), on=["cod_ibge", "ano"], how="left")
    return df.drop(columns="_rel")


@pytest.fixture(scope="module")
def treinado(tmp_path_factory):
    base = tmp_path_factory.mktemp("treino")
    dados, saida = base / "dados", base / "saida"
    dados.mkdir()
    _dataset_sintetico().to_csv(dados / "dataset_milho.csv", index=False)
    _, metricas, _ = modelar_culturas.executar(["milho"], dados, saida)
    caminho = saida / "modelos_cultura" / "modelo_milho.joblib"
    return {"artefato": joblib.load(caminho), "pasta": saida / "modelos_cultura", "metricas": metricas}


def test_artefato_guarda_metricas_fixas(treinado):
    a = treinado["artefato"]
    assert a["target"] == "anomalia_y_log"
    assert len(a["tendencia_coef"]) == 2 and a["tendencia_coef"][0] > 0  # tendência positiva recuperada
    assert a["ano_ultimo"] == 2025
    lo, hi = a["intervalo_log"]
    assert lo < 0 < hi
    assert {
        "skill_temporal_vs_tendencia",
        "skill_espacial_vs_tendencia",
        "skill_temporal_vs_mun+tend",
        "skill_temporal_sem_lag_vs_mun+tend",  # horizonte >= 2 anos (sem o rendimento do ano anterior)
    } <= a["validacao"].keys()
    perfil = a["perfil_local"]
    assert perfil["cod_ibge"].is_unique and len(perfil) == 40
    assert {"lat", "lon", "lag_ultimo", "ano_lag", "lag_medio", "anom_mun"} <= set(perfil.columns)
    assert "lag1_rel" not in perfil.columns  # o lag vem do histórico, não é constante por município


def test_metricas_incluem_baseline_e_skill(treinado):
    m = treinado["metricas"]
    assert {"MAE_tendencia", "MAE_mun+tend", "skill_vs_tendencia", "skill_vs_mun+tend"} <= set(m.columns)
    # a tendência foi aprendida: a base sem ela erraria muito mais
    assert (m["skill_vs_tendencia"] > 0).all()


def test_previsao_so_com_latitude_e_longitude(treinado):
    a = treinado["artefato"]
    alvo_mun = a["perfil_local"].iloc[5]
    r = predicao.prever_simples(a, 2026, alvo_mun["lat"] + 0.01, alvo_mun["lon"] - 0.01)
    assert r["municipio_cod_ibge"] == int(alvo_mun["cod_ibge"])
    assert r["p10_kg_ha"] < r["rendimento_predito_kg_ha"] < r["p90_kg_ha"]
    assert r["lag_origem"].startswith("observado")  # 2026 usa o rendimento de 2025 já conhecido
    assert r["solo_origem"] == "perfil Embrapa do município"
    # sanidade: perto de "tendência de 2026 + desvio histórico do município" (a regra simples de referência)
    assert r["rendimento_predito_kg_ha"] == pytest.approx(r["referencia_municipio_kg_ha"], rel=0.2)


def test_tendencia_projeta_anos_futuros(treinado):
    """Sem tendência no alvo o modelo previa a média de 2000-2025 para qualquer ano futuro."""
    a = treinado["artefato"]
    p = a["perfil_local"].iloc[0]
    # mesmo histórico (lag1_rel) nas duas chamadas, para isolar o efeito da tendência
    r26 = predicao.prever_simples(a, 2026, p["lat"], p["lon"], lag1_rel=0.0)["rendimento_predito_kg_ha"]
    r29 = predicao.prever_simples(a, 2029, p["lat"], p["lon"], lag1_rel=0.0)["rendimento_predito_kg_ha"]
    assert r29 == pytest.approx(r26 * np.exp(3 * a["tendencia_coef"][0]), rel=1e-6)
    assert r29 > r26 * 1.03  # +2%/ano durante 3 anos ~ +6%


def test_horizonte_longo_usa_media_historica_e_avisa(treinado):
    a = treinado["artefato"]
    p = a["perfil_local"].iloc[0]
    r = predicao.prever_simples(a, 2030, p["lat"], p["lon"])
    assert r["lag_origem"] == "média histórica do município"
    assert any("tendência" in av and "extrapolada" in av for av in r["avisos"])


def test_solo_informado_tem_prioridade_e_recalcula_cad(treinado):
    a = treinado["artefato"]
    p = a["perfil_local"].iloc[0]
    entrada, ctx = predicao.montar_entrada(
        a, 2026, p["lat"], p["lon"], solo={"argila": 580.0, "silte": 100.0, "prof_max_cm": 60.0}
    )
    assert entrada.loc[0, "argila"] == 580.0
    assert entrada.loc[0, "cad_mm"] != p["cad_mm"]  # recalculada com o solo do agricultor
    assert ctx["solo_origem"] == "informado pelo agricultor"
    entrada2, _ = predicao.montar_entrada(a, 2026, p["lat"], p["lon"])
    assert entrada2.loc[0, "argila"] == p["argila"]  # sem solo: usa o do município


def test_confiavel_depende_da_situacao_e_do_ganho_minimo():
    """Com histórico recente o modelo pode merecer confiança e sem ele não (e vice-versa)."""
    a = {
        "validacao": {
            "skill_temporal_vs_mun+tend": 0.10,
            "skill_temporal_sem_lag_vs_mun+tend": 0.01,  # abaixo de SKILL_MINIMO
            "skill_espacial_vs_tendencia": 0.20,
        }
    }
    assert predicao.confiavel(a, sem_historico=False) is True
    assert predicao.confiavel(a, sem_historico=True) is False
    a["validacao"]["skill_espacial_vs_tendencia"] = 0.0  # não funciona em regiões novas
    assert predicao.confiavel(a, sem_historico=False) is False
    assert predicao.confiavel({}, sem_historico=False) is False  # sem métricas: nunca "confiável"


def test_aviso_quando_modelo_nao_supera_regra_simples(treinado):
    a = dict(treinado["artefato"], validacao={"skill_temporal_vs_mun+tend": -0.1})
    p = a["perfil_local"].iloc[0]
    r = predicao.prever_simples(a, 2026, p["lat"], p["lon"])
    assert r["confiavel"] is False
    assert any("não superou" in av for av in r["avisos"])


def test_ponto_fora_da_area_e_recusado(treinado):
    with pytest.raises(ValueError, match="fora da área"):
        predicao.prever_simples(treinado["artefato"], 2026, -3.0, -60.0)  # Amazonas


def test_artefato_antigo_sem_perfil_da_erro_claro():
    antigo = {"modelo": None, "features": ["a"], "cultura": "milho"}
    with pytest.raises(ValueError, match="perfil_local"):
        predicao.montar_entrada(antigo, 2026, -22.0, -48.0)


def test_endpoint_simples_e_endpoint_antigo_respeitam_tendencia(treinado, monkeypatch):
    monkeypatch.setenv("AGRO_ML_MODELOS_DIR", str(treinado["pasta"]))
    monkeypatch.setenv("AGRO_ML_MODELO_SOJA", str(treinado["pasta"] / "nao_existe.joblib"))
    api._carregar.cache_clear()
    try:
        p = treinado["artefato"]["perfil_local"].iloc[3]
        client = TestClient(api.app)
        resp = client.post(
            "/predicoes/simples",
            json={"cultura": "milho", "ano": 2026, "latitude": p["lat"], "longitude": p["lon"]},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["p10_kg_ha"] < body["rendimento_predito_kg_ha"] < body["p90_kg_ha"]
        assert body["clima_usado"].startswith("típico")

        # solo parcial é aceito; campo desconhecido é rejeitado
        ok = client.post(
            "/predicoes/simples",
            json={
                "cultura": "milho",
                "ano": 2026,
                "latitude": p["lat"],
                "longitude": p["lon"],
                "solo": {"argila": 400},
            },
        )
        assert ok.status_code == 200 and ok.json()["solo_origem"] == "informado pelo agricultor"
        ruim = client.post(
            "/predicoes/simples",
            json={"cultura": "milho", "ano": 2026, "latitude": 1, "longitude": 1, "cor": "azul"},
        )
        assert ruim.status_code == 422

        # endpoint antigo: o artefato novo (alvo = anomalia) deve somar a tendência antes de exp()
        antigo = api.prever  # usa o mesmo helper de conversão
        assert callable(antigo)
        a = treinado["artefato"]
        assert predicao.rendimento_kg(a, 2026, 0.0) == pytest.approx(
            np.exp(np.polyval(a["tendencia_coef"], 2026.0))
        )
        assert predicao.rendimento_kg({"target": "y_log"}, 2026, np.log(3000.0)) == pytest.approx(3000.0)
    finally:
        api._carregar.cache_clear()
