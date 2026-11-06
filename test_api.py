import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api import app


def _payload(cultura="milho", ano=2020):
    datas = pd.date_range("2019-10-01", "2020-04-30", freq="D")
    clima = [
        {
            "data": data.strftime("%Y-%m-%d"),
            "temp_media": 23,
            "temp_max": 30,
            "temp_min": 17,
            "precip_mm": 4,
            "umidade_rel": 70,
            "radiacao_mj": 20,
            "vento_ms": 2,
            "ps_kpa": 95,
            "umidade_solo_raiz": 0.6,
        }
        for data in datas
    ]
    return {
        "cultura": cultura,
        "ano": ano,
        "municipio": {"cod_ibge": 3500105, "latitude": -22.3, "longitude": -48.3},
        "solo": {
            "ph_h2o": 5.5,
            "carbono_organico": 15,
            "ctc": 80,
            "saturacao_bases": 50,
            "aluminio": 2,
            "argila": 350,
            "silte": 250,
            "prof_max_cm": 100,
        },
        "clima_diario": clima,
        "lag1_rel": 0.0,
    }


def test_health_and_catalogo_expose_models():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["modelos_disponiveis"] >= 9
    catalogo = client.get("/modelos")
    assert catalogo.status_code == 200
    assert {item["cultura"] for item in catalogo.json()} >= {"soja", "milho"}


def test_predicao_de_dados_brutos():
    response = TestClient(app).post("/predicoes", json=_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cultura"] == "milho"
    assert np.isfinite(body["rendimento_predito_kg_ha"])
    assert body["rendimento_predito_kg_ha"] >= 0


def test_predicao_soja_usa_artefato_legado():
    response = TestClient(app).post("/predicoes", json=_payload("soja"))
    assert response.status_code == 200, response.text
    assert response.json()["cultura"] == "soja"


def test_predicao_rejeita_cultura_desconhecida():
    payload = _payload("cacau")
    response = TestClient(app).post("/predicoes", json=payload)
    assert response.status_code == 422
