"""Testes do pipeline. Rode:  python -m pytest -q test_pipeline.py   (ou: python test_pipeline.py)"""

import numpy as np
import pandas as pd

import alvo
import avaliar
import build_dataset
import clima
import config as C
import modelar_culturas


def _diario_sintetico(seed, anos=("1999-01-01", "2025-12-31"), fator_chuva=None):
    rng = np.random.default_rng(seed)
    datas = pd.date_range(*anos)
    doy = datas.dayofyear.values
    saz = np.cos(2 * np.pi * (doy - 15) / 365)
    # variabilidade interanual de chuva (anos secos e úmidos)
    mult = pd.Series(
        rng.lognormal(0, 0.35, datas.year.nunique()), index=sorted(datas.year.unique())
    )
    chuva = (
        np.where(
            rng.random(len(datas)) < (0.40 + 0.25 * saz),
            rng.exponential(9, len(datas)),
            0,
        )
        * mult.loc[datas.year].values
    )
    return pd.DataFrame(
        {
            "data": datas,
            "temp_media": 22 + 3.5 * saz + rng.normal(0, 1, len(datas)),
            "temp_max": 29 + 4 * saz + rng.normal(0, 1.5, len(datas)),
            "temp_min": 16 + 4.5 * saz + rng.normal(0, 1.5, len(datas)),
            "precip_mm": chuva,
            "umidade_rel": np.clip(
                72 + 8 * saz + rng.normal(0, 6, len(datas)), 30, 100
            ),
            "radiacao_mj": np.clip(18 + 5 * saz + rng.normal(0, 3, len(datas)), 4, 30),
            "vento_ms": np.clip(2 + rng.normal(0, 0.5, len(datas)), 0.3, 6),
            "ps_kpa": 94.0,
        }
    )


def test_eto_plausivel_sp():
    d = clima.preparar_diario(_diario_sintetico(0))
    eto = clima.eto_fao56(d, -22.3, 600)
    verao, inverno = (
        eto[d.data.dt.month.isin([12, 1, 2])].mean(),
        eto[d.data.dt.month.isin([6, 7, 8])].mean(),
    )
    assert 3.5 < verao < 6.5 and 1.5 < inverno < 4.0 and verao > inverno


def test_balanco_hidrico_seca_reduz_eta():
    eta = clima.balanco_hidrico(np.zeros(60), np.full(60, 5.0), 80.0)
    assert eta[-1] < eta[0] and eta.min() >= 0


def test_janelas_sem_cobertura_viram_nan():
    d = clima.preparar_diario(_diario_sintetico(1, anos=("2005-01-01", "2010-12-31")))
    f = clima.features_cultura(d, "soja", range(2005, 2008), -22.3, 600, 90)
    assert f.loc[f.ano == 2005, "floracao_prec"].isna().all()  # falta dez/2004
    assert f.loc[f.ano == 2007, "floracao_prec"].notna().all()


def test_limpeza_remove_quebra_de_serie_e_outliers():
    base = dict(
        localidade="x",
        area_colhida=500.0,
        area_plantada=500.0,
        producao=1250.0,
        produto="laranja",
    )
    df = pd.DataFrame(
        [
            {
                **base,
                "ano": 2000,
                "rendimento": 160007.0,
            },  # frutos/ha (quebra de série)
            {**base, "ano": 2001, "rendimento": 25000.0},
            {**base, "ano": 2002, "rendimento": 26000.0},
            {**base, "ano": 2003, "rendimento": 318628.0},  # absurdo
            {**base, "ano": 2004, "rendimento": 27000.0},
        ]
    )
    limpo, rel = alvo.limpar_pam(df)
    assert set(limpo["ano"]) == {2001, 2002, 2004}


def test_tendencia_so_usa_treino():
    anos = pd.Series(np.repeat(np.arange(2000, 2010), 5))
    y = pd.Series(0.02 * (anos - 2000) + 7.0)
    t = alvo.Tendencia().fit(anos, y)
    assert abs(t.predict([2012])[0] - (0.02 * 12 + 7.0)) < 1e-6


def test_lag_relativo_nao_vaza_futuro():
    df = pd.DataFrame(
        {
            "cod_ibge": [1, 1, 1, 2, 2, 2],
            "produto": "soja",
            "ano": [2000, 2001, 2002] * 2,
            "y_log": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        }
    )
    out = build_dataset.adicionar_lag(df).sort_values(["cod_ibge", "ano"])
    assert out[out.ano == 2000]["lag1_rel"].isna().all()
    # lag de 2001 do município 1 = rel de 2000 = 1.0 - mediana(1.0,1.5)=-0.25
    assert (
        abs(out[(out.cod_ibge == 1) & (out.ano == 2001)]["lag1_rel"].iloc[0] + 0.25)
        < 1e-9
    )


def test_avaliacao_recupera_sinal_sintetico():
    """Se o rendimento DEPENDE do déficit hídrico na floração, o modelo com features por janela deve bater a baseline
    em regiões novas e em anos futuros. Garante que o método de avaliação não está estruturalmente enviesado."""
    rng = np.random.default_rng(42)
    linhas = []
    celulas = [(-20.5 - 0.5 * i, -48 - 0.625 * j) for i in range(4) for j in range(3)]
    for c, (la, lo) in enumerate(celulas):
        d = clima.preparar_diario(_diario_sintetico(100 + c))
        for m in range(6):
            cod = c * 10 + m
            cad = float(rng.uniform(60, 140))
            f = clima.features_cultura(d, "soja", range(2000, 2026), la, 600, cad)
            mun_ef = rng.normal(0, 0.08)
            f["cod_ibge"], f["produto"], f["lat"], f["lon"], f["cad_mm"] = (
                cod,
                "soja",
                la + rng.normal(0, 0.1),
                lo + rng.normal(0, 0.1),
                cad,
            )
            f["localidade"] = f"m{cod}"
            tend = 0.015 * (f["ano"] - 2000)
            f["y_log"] = (
                np.log(2600)
                + tend
                + mun_ef
                - 0.006 * f["floracao_defic"]
                - 0.004 * f["enchimento_defic"]
                + rng.normal(0, 0.06, len(f))
            )
            f["rendimento"] = np.exp(f["y_log"])
            f["area_colhida"] = 800.0
            f["peso"] = np.sqrt(800.0)
            linhas.append(f)
    df = pd.concat(linhas, ignore_index=True).dropna(subset=["floracao_defic"])
    res, imp, feats = avaliar.avaliar_cultura(df, "soja", importancia=True)
    esp = res[res.validacao.str.startswith("espacial")].iloc[0]
    tmp = res[res.validacao.str.startswith("temporal")].iloc[0]
    assert esp["skill_vs_tendencia"] > 0.15 and tmp["skill_vs_tendencia"] > 0.15
    assert any(
        ("defic" in k) or ("estresse" in k) for k in imp.head(4).index
    )  # o fator verdadeiro aparece no topo
    print(res.round(3).to_string(index=False))
    print(imp.head(5))


def test_selecao_modelo_prioriza_mae_temporal():
    metricas = pd.DataFrame(
        [
            {"modelo": "ElasticNet", "validacao": "temporal (anos futuros)", "MAE_kg_ha": 100},
            {"modelo": "RandomForest", "validacao": "temporal (anos futuros)", "MAE_kg_ha": 80},
            {"modelo": "HistGradientBoosting", "validacao": "espacial (regiões novas)", "MAE_kg_ha": 10},
        ]
    )
    assert modelar_culturas.selecionar_modelo(metricas) == "RandomForest"


def test_analise_nao_conta_ausencia_de_colunas_irrelevantes():
    df = pd.DataFrame(
        {
            "produto": ["trigo", "trigo", "trigo"],
            "cod_ibge": [1, 2, 3],
            "localidade": ["a", "b", "c"],
            "ano": [2020, 2021, 2022],
            "rendimento": [2000.0, 2200.0, 2100.0],
            "y_log": np.log([2000.0, 2200.0, 2100.0]),
            "peso": [1.0, 1.0, 1.0],
            "feature": [1.0, 2.0, np.nan],
            "janela_inexistente_tmax": [np.nan, np.nan, np.nan],
        }
    )
    resumo = modelar_culturas.analisar_dados(df, "trigo")
    assert resumo["linhas_com_ausencia_features"] == 1


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print("ok", nome)
