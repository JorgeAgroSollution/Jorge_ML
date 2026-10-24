"""Avaliação honesta por cultura: validação espacial (regiões novas) e temporal (anos futuros), contra baseline.

Uso:
    python avaliar.py --dados data/processed/dataset_completo.csv
    python avaliar.py --dados /caminho/df_merged.csv --legado        # CSV antigo (clima anual, id = localidade)

O que muda em relação ao RandomForest com KFold aleatório:
  * alvo = anomalia do log-rendimento em relação à tendência tecnológica (ajustada só no treino de cada fold);
  * split espacial: clusters de lat/lon inteiros ficam de fora (o clima NASA tem resolução de ~50 km);
  * split temporal: expanding window (treina no passado, testa em 3 anos seguintes);
  * métricas em kg/ha e skill sobre o baseline (só tendência) — se skill <= 0 o modelo NÃO agrega valor;
  * quantis P10/P50/P90 para risco ("ano ruim") e cobertura empírica do intervalo.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

import alvo
import config as C

warnings.filterwarnings("ignore")


def selecionar_features(
    df: pd.DataFrame, com_coordenadas: bool = False, com_lag: bool = True
) -> list[str]:
    excl = set(C.COLS_VAZAMENTO) | set(C.COLS_REDUNDANTES)
    if not com_coordenadas:
        excl |= {"lat", "lon", "latitude", "longitude"}
    if not com_lag:
        excl |= {"lag1_rel"}
    num = df.select_dtypes(include="number").columns
    feats = [
        c
        for c in num
        if c not in excl and df[c].notna().mean() > 0.5 and df[c].nunique() > 1
    ]
    # remove duplicatas exatas de coluna (ex.: lat == latitude_x já tratado) e constantes por cultura
    return list(dict.fromkeys(feats))


def _hgb(loss="squared_error", q=None):
    kw = dict(
        loss=loss,
        max_depth=3,
        learning_rate=0.05,
        max_iter=250,
        min_samples_leaf=25,
        l2_regularization=2.0,
        random_state=0,
    )
    if q is not None:
        kw["quantile"] = q
    return HistGradientBoostingRegressor(**kw)


def _ajustar_prever(df, feats, tr, te):
    """Treina no índice `tr`, prevê em `te`. Devolve DataFrame com baselines e previsões (kg/ha)."""
    dtr, dte = df.iloc[tr], df.iloc[te]
    tend = alvo.Tendencia().fit(dtr["ano"], dtr["y_log"])
    y_anom = dtr["y_log"].values - tend.predict(dtr["ano"])
    t_te = tend.predict(dte["ano"])
    w = dtr["peso"].values
    out = pd.DataFrame(index=dte.index)
    out["y"] = dte["rendimento"].values
    out["y_log"] = dte["y_log"].values
    out["base_tend"] = np.exp(t_te)
    # baseline mais forte (só faz sentido no split temporal): tendência + anomalia média histórica do município
    mun_anom = pd.Series(y_anom, index=dtr["_mun"].values).groupby(level=0).mean()
    out["base_mun"] = np.exp(t_te + dte["_mun"].map(mun_anom).fillna(0).values)
    X_tr, X_te = dtr[feats], dte[feats]
    m = _hgb().fit(X_tr, y_anom, sample_weight=w)
    out["modelo"] = np.exp(t_te + m.predict(X_te))
    # Quantis P10/P90 + calibração conformal (CQR): 20% dos MUNICÍPIOS do treino ficam de calibração
    rng = np.random.default_rng(0)
    muns = dtr["_mun"].unique()
    cal_m = set(rng.choice(muns, size=max(1, int(0.2 * len(muns))), replace=False))
    is_cal = dtr["_mun"].isin(cal_m).values
    q = {}
    for alfa in (0.1, 0.9):
        mq = _hgb("quantile", alfa).fit(
            X_tr[~is_cal], y_anom[~is_cal], sample_weight=w[~is_cal]
        )
        q[alfa] = (mq.predict(X_tr[is_cal]), mq.predict(X_te))
    lo_c, hi_c = np.minimum(q[0.1][0], q[0.9][0]), np.maximum(q[0.1][0], q[0.9][0])
    score = np.maximum(lo_c - y_anom[is_cal], y_anom[is_cal] - hi_c)
    n_c = len(score)
    Q = np.quantile(score, min(1.0, 0.8 * (1 + 1 / n_c))) if n_c >= 20 else 0.0
    lo, hi = np.minimum(q[0.1][1], q[0.9][1]) - Q, np.maximum(q[0.1][1], q[0.9][1]) + Q
    out["p10"], out["p90"] = np.exp(t_te + lo), np.exp(t_te + hi)
    return out, m, X_te, dte["y_log"].values - t_te


def _metricas(res: pd.DataFrame) -> dict:
    mae = lambda c: float(np.mean(np.abs(res["y"] - res[c])))
    y_an = np.log(res["y"]) - np.log(res["base_tend"])
    p_an = np.log(res["modelo"]) - np.log(res["base_tend"])
    r2_anom = 1 - np.sum((y_an - p_an) ** 2) / np.sum((y_an - y_an.mean()) ** 2)
    p10, p90 = (
        res["p10"].where(res["p10"] <= res["p90"], res["p90"]),
        res["p90"].where(res["p10"] <= res["p90"], res["p10"]),
    )
    return {
        "n_teste": len(res),
        "MAE_tendencia": mae("base_tend"),
        "MAE_mun+tend": mae("base_mun"),
        "MAE_modelo": mae("modelo"),
        "skill_vs_tendencia": 1 - mae("modelo") / mae("base_tend"),
        "skill_vs_mun+tend": 1 - mae("modelo") / mae("base_mun"),
        "R2_anomalia": float(r2_anom),
        "abaixo_P10": float(np.mean(res["y"] < p10)),  # ideal ~0,10
        "dentro_P10_P90": float(
            np.mean((res["y"] >= p10) & (res["y"] <= p90))
        ),  # ideal ~0,80
    }


def avaliar_cultura(
    df: pd.DataFrame,
    cultura: str,
    com_coordenadas=False,
    com_lag=True,
    k_espacial=8,
    importancia=True,
):
    df = df[df["produto"] == cultura].reset_index(drop=True)
    df["_mun"] = df["cod_ibge"] if "cod_ibge" in df else df["localidade"]
    feats = selecionar_features(df, com_coordenadas, com_lag)
    # grupos espaciais: clusters de lat/lon (municípios próximos compartilham a célula climática)
    mun = df.groupby("_mun")[["lat", "lon"]].first()
    k = min(k_espacial, len(mun))
    mun["_g"] = KMeans(k, n_init=10, random_state=0).fit_predict(mun[["lat", "lon"]])
    grupos = df["_mun"].map(mun["_g"]).values

    linhas, imp_total = [], []
    # --- espacial ---
    pred = []
    for tr, te in GroupKFold(min(5, k)).split(df, groups=grupos):
        out, m, Xte, yte = _ajustar_prever(df, feats, tr, te)
        pred.append(out)
        if importancia:
            pi = permutation_importance(
                m,
                Xte,
                yte,
                n_repeats=3,
                random_state=0,
                scoring="neg_mean_absolute_error",
            )
            imp_total.append(pd.Series(pi.importances_mean, index=feats))
    r = _metricas(pd.concat(pred))
    r.update(cultura=cultura, validacao="espacial (regiões novas)")
    linhas.append(r)
    # --- temporal (expanding window) ---
    pred = []
    anos = sorted(df["ano"].unique())
    for ano_ini in (anos[-1] - 8, anos[-1] - 5, anos[-1] - 2):
        tr, te = (
            np.where(df["ano"] < ano_ini)[0],
            np.where((df["ano"] >= ano_ini) & (df["ano"] < ano_ini + 3))[0],
        )
        if len(tr) < 200 or len(te) < 30:
            continue
        out, *_ = _ajustar_prever(df, feats, tr, te)
        pred.append(out)
    if pred:
        r = _metricas(pd.concat(pred))
        r.update(cultura=cultura, validacao="temporal (anos futuros)")
        linhas.append(r)
    imp = (
        pd.concat(imp_total, axis=1).mean(axis=1).sort_values(ascending=False)
        if imp_total
        else pd.Series(dtype=float)
    )
    return pd.DataFrame(linhas), imp, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True)
    ap.add_argument(
        "--legado",
        action="store_true",
        help="CSV antigo: sem código IBGE, limpa e usa localidade como id",
    )
    ap.add_argument("--culturas", nargs="*")
    ap.add_argument("--com-coordenadas", action="store_true")
    ap.add_argument("--sem-lag", action="store_true")
    ap.add_argument(
        "--descartar-repetidos",
        action="store_true",
        help="exclui (em vez de rebaixar o peso) rendimentos repetidos >=3 anos",
    )
    ap.add_argument(
        "--sem-limpeza",
        action="store_true",
        help="não aplica alvo.limpar_pam (para comparar)",
    )
    ap.add_argument("--saida", default=str(C.DIR_RESULTADOS))
    a = ap.parse_args()

    df = pd.read_csv(a.dados)
    if a.legado or "y_log" not in df.columns:
        if a.sem_limpeza:
            df = df.assign(
                y_log=np.log(df["rendimento"]), peso=np.sqrt(df["area_colhida"])
            )
        else:
            df, rel = alvo.limpar_pam(df, descartar_repetidos=a.descartar_repetidos)
            print("=== Relatório de limpeza do alvo ===\n", rel.to_string(), "\n")
    culturas = a.culturas or [c for c in C.CULTURAS if c in df["produto"].unique()]
    res, imps = [], {}
    for c in culturas:
        r, imp, feats = avaliar_cultura(df, c, a.com_coordenadas, not a.sem_lag)
        res.append(r)
        imps[c] = imp
    res = pd.concat(res, ignore_index=True)
    cols = [
        "cultura",
        "validacao",
        "n_teste",
        "MAE_tendencia",
        "MAE_mun+tend",
        "MAE_modelo",
        "skill_vs_tendencia",
        "skill_vs_mun+tend",
        "R2_anomalia",
        "abaixo_P10",
        "dentro_P10_P90",
    ]
    res = res[cols]
    import os

    os.makedirs(a.saida, exist_ok=True)
    res.to_csv(f"{a.saida}/metricas.csv", index=False)
    pd.DataFrame(imps).to_csv(f"{a.saida}/importancia_permutacao.csv")
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print("\nTop-5 features (importância por permutação, ΔMAE log) por cultura:")
    for c, s in imps.items():
        print(f"  {c}: " + ", ".join(f"{k}({v:.3f})" for k, v in s.head(5).items()))


if __name__ == "__main__":
    main()
