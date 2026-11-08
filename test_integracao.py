"""Testes do que o Jorge consome: lote, exportação das previsões, laudo do IAC e versão do modelo. Sem rede.

    uv run pytest -q test_integracao.py
"""

import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
import config as C
import exportar_previsoes
import modelar_culturas
import predicao
from test_predicao import _dataset_sintetico


@pytest.fixture(scope="module")
def treinado(tmp_path_factory):
    base = tmp_path_factory.mktemp("treino_integracao")
    dados, saida = base / "dados", base / "saida"
    dados.mkdir()
    _dataset_sintetico(n_mun=12).to_csv(dados / "dataset_milho.csv", index=False)
    modelar_culturas.executar(["milho"], dados, saida)
    pasta = saida / "modelos_cultura"
    return {"artefato": joblib.load(pasta / "modelo_milho.joblib"), "pasta": pasta, "base": base}


@pytest.fixture
def cliente(treinado, monkeypatch):
    monkeypatch.setenv("AGRO_ML_MODELOS_DIR", str(treinado["pasta"]))
    monkeypatch.setenv("AGRO_ML_MODELO_SOJA", str(treinado["pasta"] / "nao_existe.joblib"))
    api._carregar.cache_clear()
    yield TestClient(api.app)
    api._carregar.cache_clear()


def test_laudo_do_iac_vira_unidades_do_modelo():
    s = predicao.solo_do_laudo({"ph_cacl2": 5.0, "materia_organica_g_dm3": 30.0, "ctc_mmolc_dm3": 80.0})
    assert s["ph_h2o"] == pytest.approx(5.0 + C.PH_H2O_MENOS_CACL2)
    assert s["carbono_organico"] == pytest.approx(30.0 / 1.724)
    assert s["ctc"] == pytest.approx(8.0)  # 80 mmolc/dm3 = 8 cmolc/dm3, a unidade dos dados de treino
    assert "ph_cacl2" not in s and "ctc_mmolc_dm3" not in s


def test_campo_na_unidade_do_modelo_tem_prioridade_sobre_o_laudo():
    s = predicao.solo_do_laudo({"ph_h2o": 6.1, "ph_cacl2": 5.0, "ctc": 9.0, "ctc_mmolc_dm3": None})
    assert s == {"ph_h2o": 6.1, "ctc": 9.0}
    assert predicao.solo_do_laudo(None) is None


def test_avisos_tem_codigo_na_mesma_ordem(treinado):
    a = treinado["artefato"]
    p = a["perfil_local"].iloc[0]
    r = predicao.prever_simples(a, a["ano_ultimo"] + 5, p["lat"], p["lon"])
    assert r["avisos_codigos"][:2] == [predicao.AVISO_SEM_LAG, predicao.AVISO_EXTRAPOLACAO]
    assert len(r["avisos_codigos"]) == len(r["avisos"])
    perto = predicao.prever_simples(a, a["ano_ultimo"] + 1, p["lat"], p["lon"])
    assert predicao.AVISO_SEM_LAG not in perto["avisos_codigos"]


def test_aviso_de_distancia_tem_codigo(treinado):
    a = treinado["artefato"]
    p = a["perfil_local"].iloc[0]
    # ~0,6 grau de latitude ~ 67 km: entre o aviso (50 km) e a recusa (150 km), se não houver vizinho mais perto
    perfil = a["perfil_local"].iloc[[0]]
    sozinho = {**a, "perfil_local": perfil}
    r = predicao.prever_simples(sozinho, 2026, p["lat"] + 0.6, p["lon"])
    assert predicao.AVISO_DISTANCIA in r["avisos_codigos"]


def test_versao_do_artefato_e_o_sha256_curto(tmp_path):
    arq = tmp_path / "modelo_x.joblib"
    arq.write_bytes(b"abc")
    assert api.versao_artefato(arq) == "ba7816bf8f01"  # sha256("abc")
    arq.write_bytes(b"abcd")
    assert api.versao_artefato(arq) != "ba7816bf8f01"


def test_respostas_trazem_versao_e_codigos(cliente, treinado):
    a = treinado["artefato"]
    p = a["perfil_local"].iloc[2]
    versao = api.versao_artefato(treinado["pasta"] / "modelo_milho.joblib")
    modelos = cliente.get("/modelos").json()
    assert modelos[0]["versao"] == versao and modelos[0]["ano_ultimo"] == a["ano_ultimo"]

    resp = cliente.post(
        "/predicoes/simples",
        json={
            "cultura": "milho",
            "ano": 2026,
            "latitude": p["lat"],
            "longitude": p["lon"],
            "solo": {"ph_cacl2": 5.2, "materia_organica_g_dm3": 25, "ctc_mmolc_dm3": 70},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["versao_modelo"] == versao and body["ano_ultimo"] == a["ano_ultimo"]
    assert body["avisos_codigos"] == []
    assert body["solo_origem"] == "informado pelo agricultor"


def test_lote_isola_o_erro_de_cada_cultura(cliente, treinado):
    p = treinado["artefato"]["perfil_local"].iloc[1]
    resp = cliente.post(
        "/predicoes/lote",
        json={"culturas": ["milho", "cacau", "soja"], "ano": 2026, "latitude": p["lat"], "longitude": p["lon"]},
    )
    assert resp.status_code == 200, resp.text
    milho, cacau, soja = resp.json()["resultados"]
    assert milho["cultura"] == "milho" and milho["erro"] is None
    sozinho = cliente.post(
        "/predicoes/simples",
        json={"cultura": "milho", "ano": 2026, "latitude": p["lat"], "longitude": p["lon"]},
    ).json()
    assert milho["previsao"] == sozinho
    assert cacau["previsao"] is None and "não suportada" in cacau["erro"]
    assert soja["previsao"] is None and "não encontrado" in soja["erro"]  # sem artefato de soja na pasta


def test_lote_fora_da_area_vira_erro_do_item(cliente):
    resp = cliente.post("/predicoes/lote", json={"culturas": ["milho"], "ano": 2026, "latitude": 1, "longitude": 1})
    assert resp.status_code == 200
    assert "fora da área" in resp.json()["resultados"][0]["erro"]


def test_exportacao_usa_o_mesmo_calculo_da_api(treinado, tmp_path):
    a = treinado["artefato"]
    perfil = a["perfil_local"]
    municipios = tmp_path / "municipios_ref.csv"
    pd.DataFrame(
        {"cod_ibge": perfil["cod_ibge"], "municipio": [f"Cidade {i}" for i in range(len(perfil))]}
    ).to_csv(municipios, index=False)

    doc = exportar_previsoes.exportar(treinado["pasta"], municipios)
    anos = [a["ano_ultimo"] + i for i in (1, 2, 3)]
    assert len(doc["linhas"]) == len(perfil) * 3
    assert sorted({linha["ano"] for linha in doc["linhas"]}) == anos
    assert doc["modelos"]["milho"]["versao"] == api.versao_artefato(treinado["pasta"] / "modelo_milho.joblib")

    linha = doc["linhas"][4]
    m = perfil[perfil["cod_ibge"] == linha["cod_ibge"]].iloc[0]
    r = predicao.prever_simples(a, linha["ano"], m["lat"], m["lon"])
    assert linha["kg_ha"] == round(r["rendimento_predito_kg_ha"])
    assert linha["p10_kg_ha"] == round(r["p10_kg_ha"]) and linha["p90_kg_ha"] == round(r["p90_kg_ha"])
    assert linha["referencia_kg_ha"] == round(r["referencia_municipio_kg_ha"])
    assert linha["confiavel"] == r["confiavel"]
    assert linha["municipio"].startswith("Cidade ")
    assert "extrapolado" not in linha  # 3 anos após o último dado: nunca extrapola

    # 1º ano usa o rendimento observado; os seguintes, a média do município
    origens = {(linha["ano"], linha["lag_origem"]) for linha in doc["linhas"]}
    assert origens == {(anos[0], "observado"), (anos[1], "media"), (anos[2], "media")}


def test_versao_da_exportacao_so_muda_com_o_modelo(treinado, tmp_path):
    municipios = tmp_path / "m.csv"
    pd.DataFrame({"cod_ibge": [1], "municipio": ["X"]}).to_csv(municipios, index=False)
    d1 = exportar_previsoes.exportar(treinado["pasta"], municipios)
    d2 = exportar_previsoes.exportar(treinado["pasta"], municipios)
    assert d1["versao"] == d2["versao"]
    assert exportar_previsoes.exportar(treinado["pasta"], municipios, n_anos=5)["versao"] != d1["versao"]
    longe = [linha for linha in exportar_previsoes.exportar(treinado["pasta"], municipios, 5)["linhas"]]
    assert any(linha.get("extrapolado") for linha in longe)

    saida = tmp_path / "previsoes.json"
    tamanho = exportar_previsoes.gravar(d1, saida)
    assert tamanho == saida.stat().st_size
    lido = json.loads(saida.read_text(encoding="utf-8"))
    assert lido["versao"] == d1["versao"] and lido["unidade"] == "kg/ha"
