"""Compara, treina e persiste modelos de rendimento por cultura.

Uso:
    uv run python modelar_culturas.py
    uv run python modelar_culturas.py --culturas milho cafe
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C
import unidades
from avaliar import selecionar_features

MODELOS = ("ElasticNet", "RandomForest", "HistGradientBoosting")


def _pipeline(nome: str) -> Pipeline:
    passos = [("imputacao", SimpleImputer(strategy="median"))]
    if nome == "ElasticNet":
        passos.extend(
            [
                ("escala", StandardScaler()),
                ("modelo", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10_000, random_state=42)),
            ]
        )
    elif nome == "RandomForest":
        passos.append(
            (
                "modelo",
                RandomForestRegressor(
                    n_estimators=50,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )
    elif nome == "HistGradientBoosting":
        passos.append(
            (
                "modelo",
                HistGradientBoostingRegressor(
                    max_iter=100,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    learning_rate=0.05,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            )
        )
    else:
        raise ValueError(f"Modelo desconhecido: {nome}")
    return Pipeline(passos)


def analisar_dados(df: pd.DataFrame, cultura: str) -> dict:
    """Resume qualidade, cobertura e distribuição do dataset de uma cultura."""
    n_municipios = df["cod_ibge"].nunique() if "cod_ibge" in df else df["localidade"].nunique()
    features = selecionar_features(df, com_coordenadas=False, com_lag=True)
    ausentes_features = df[features].isna()
    return {
        "cultura": cultura,
        "linhas": len(df),
        "colunas": len(df.columns),
        "features_validas": len(features),
        "ano_min": int(df["ano"].min()),
        "ano_max": int(df["ano"].max()),
        "n_anos": int(df["ano"].nunique()),
        "n_municipios": int(n_municipios),
        "linhas_com_ausencia_features": int(ausentes_features.any(axis=1).sum()),
        "celulas_ausentes_features": int(ausentes_features.sum().sum()),
        "frac_solo_imputado": float(df["solo_imputado"].mean())
        if "solo_imputado" in df
        else np.nan,
        "rendimento_mediano": float(df["rendimento"].median()),
        "rendimento_medio": float(df["rendimento"].mean()),
        "rendimento_desvio": float(df["rendimento"].std()),
        "desvio_y_log": float(df["y_log"].std()),
    }


def carregar_cultura(cultura: str, diretorio: Path = C.DIR_PROC) -> pd.DataFrame:
    caminho = diretorio / f"dataset_{cultura}.csv"
    if not caminho.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {caminho}")
    df = unidades.para_modelagem(pd.read_csv(caminho))
    obrigatorias = {"ano", "produto", "rendimento", "y_log", "peso", "cod_ibge"}
    faltantes = obrigatorias - set(df.columns)
    if faltantes:
        raise ValueError(f"{caminho} não contém colunas obrigatórias: {sorted(faltantes)}")
    return df[df["produto"].eq(cultura)].dropna(subset=["rendimento", "y_log"]).reset_index(drop=True)


def _grupos_espaciais(df: pd.DataFrame, k_max: int = 8) -> tuple[np.ndarray, int]:
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("Validação espacial requer as colunas lat e lon.")
    municipios = df.groupby("cod_ibge")[["lat", "lon"]].first().dropna()
    k = min(k_max, len(municipios))
    if k < 2:
        raise ValueError("São necessários pelo menos dois municípios para validação espacial.")
    grupos_mun = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(
        municipios[["lat", "lon"]]
    )
    return df["cod_ibge"].map(pd.Series(grupos_mun, index=municipios.index)).to_numpy(), k


def _metricas(y_log: pd.Series, pred_log: np.ndarray) -> dict:
    real = np.exp(np.asarray(y_log))
    previsto = np.exp(pred_log)
    smape = np.mean(
        2 * np.abs(real - previsto) / np.maximum(np.abs(real) + np.abs(previsto), 1e-9)
    )
    return {
        "MAE_kg_ha": float(mean_absolute_error(real, previsto)),
        "RMSE_kg_ha": float(mean_squared_error(real, previsto) ** 0.5),
        "R2_log": float(r2_score(y_log, pred_log)),
        "sMAPE": float(smape),
    }


def _avaliar_split(
    df: pd.DataFrame,
    features: list[str],
    nome_modelo: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    validacao: str,
) -> list[dict]:
    linhas = []
    for fold, (treino, teste) in enumerate(splits, 1):
        modelo = _pipeline(nome_modelo)
        modelo.fit(
            df.iloc[treino][features],
            df.iloc[treino]["y_log"],
            modelo__sample_weight=df.iloc[treino]["peso"].to_numpy(),
        )
        pred = modelo.predict(df.iloc[teste][features])
        met = _metricas(df.iloc[teste]["y_log"], pred)
        met.update(
            cultura=str(df.iloc[0]["produto"]),
            modelo=nome_modelo,
            validacao=validacao,
            fold=fold,
            n_treino=len(treino),
            n_teste=len(teste),
        )
        linhas.append(met)
    return linhas


def _splits_temporais(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    anos = sorted(df["ano"].unique())
    splits = []
    for inicio in (anos[-1] - 5, anos[-1] - 2):
        treino = np.flatnonzero(df["ano"].to_numpy() < inicio)
        teste = np.flatnonzero(
            (df["ano"].to_numpy() >= inicio) & (df["ano"].to_numpy() < inicio + 3)
        )
        if len(treino) >= 100 and len(teste) >= 30:
            splits.append((treino, teste))
    return splits


def avaliar_modelos(df: pd.DataFrame, cultura: str) -> tuple[pd.DataFrame, list[str]]:
    """Compara os modelos em validações espacial e temporal."""
    features = selecionar_features(df, com_coordenadas=False, com_lag=True)
    if not features:
        raise ValueError(f"Nenhuma feature válida para {cultura}.")
    grupos, n_grupos = _grupos_espaciais(df)
    n_splits = min(2, n_grupos)
    espaciais = list(GroupKFold(n_splits=n_splits).split(df, groups=grupos))
    temporais = _splits_temporais(df)
    linhas = []
    for nome in MODELOS:
        linhas.extend(
            _avaliar_split(df, features, nome, espaciais, "espacial (regiões novas)")
        )
        if temporais:
            linhas.extend(_avaliar_split(df, features, nome, temporais, "temporal (anos futuros)"))
    return pd.DataFrame(linhas), features


def selecionar_modelo(metricas: pd.DataFrame) -> str:
    """Escolhe o menor MAE temporal; falha explicitamente sem validação temporal."""
    temporais = metricas[metricas["validacao"].eq("temporal (anos futuros)")]
    if temporais.empty:
        raise ValueError("Não foi possível selecionar modelo: validação temporal indisponível.")
    resumo = temporais.groupby("modelo", as_index=False)["MAE_kg_ha"].mean()
    return str(resumo.sort_values(["MAE_kg_ha", "modelo"]).iloc[0]["modelo"])


def treinar_finais(
    df: pd.DataFrame, cultura: str, features: list[str], nome_modelo: str, diretorio: Path
) -> pd.DataFrame:
    diretorio.mkdir(parents=True, exist_ok=True)
    modelo = _pipeline(nome_modelo)
    modelo.fit(df[features], df["y_log"], modelo__sample_weight=df["peso"].to_numpy())
    pred = np.exp(modelo.predict(df[features]))
    artefato = {
        "modelo": modelo,
        "modelo_nome": nome_modelo,
        "cultura": cultura,
        "features": features,
        "target": "y_log",
        "unidade_predicao": "kg/ha",
        "n_linhas_treino": len(df),
    }
    joblib.dump(artefato, diretorio / f"modelo_{cultura}.joblib")
    return pd.DataFrame(
        {
            "cultura": cultura,
            "cod_ibge": df["cod_ibge"].to_numpy(),
            "localidade": df["localidade"].to_numpy(),
            "ano": df["ano"].to_numpy(),
            "rendimento_observado_kg_ha": df["rendimento"].to_numpy(),
            "rendimento_predito_kg_ha": pred,
            "modelo": nome_modelo,
            "tipo_predicao": "ajuste_completo",
        }
    )


def executar(
    culturas: list[str] | None = None,
    diretorio_dados: Path = C.DIR_PROC,
    diretorio_resultados: Path = C.DIR_RESULTADOS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    culturas = culturas or [c for c in C.CULTURAS if c != "soja"]
    analises, metricas, previsoes = [], [], []
    for cultura in culturas:
        if cultura == "soja":
            continue
        df = carregar_cultura(cultura, diretorio_dados)
        analises.append(analisar_dados(df, cultura))
        met, features = avaliar_modelos(df, cultura)
        escolhido = selecionar_modelo(met)
        met["modelo_selecionado"] = met["modelo"].eq(escolhido)
        metricas.append(met)
        previsoes.append(treinar_finais(df, cultura, features, escolhido, diretorio_resultados / "modelos_cultura"))
    analise_df = pd.DataFrame(analises)
    metricas_df = pd.concat(metricas, ignore_index=True)
    previsoes_df = pd.concat(previsoes, ignore_index=True)
    diretorio_resultados.mkdir(parents=True, exist_ok=True)
    analise_df.to_csv(diretorio_resultados / "relatorio_analise_culturas.csv", index=False)
    metricas_df.to_csv(diretorio_resultados / "metricas_modelos.csv", index=False)
    previsoes_df.to_csv(diretorio_resultados / "previsoes_culturas.csv", index=False)
    return analise_df, metricas_df, previsoes_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--culturas", nargs="*", choices=[c for c in C.CULTURAS if c != "soja"])
    parser.add_argument("--dados", type=Path, default=C.DIR_PROC)
    parser.add_argument("--saida", type=Path, default=C.DIR_RESULTADOS)
    args = parser.parse_args()
    analise, metricas, _ = executar(args.culturas, args.dados, args.saida)
    resumo = (
        metricas[metricas["modelo_selecionado"]]
        .groupby(["cultura", "modelo", "validacao"], as_index=False)["MAE_kg_ha"]
        .mean()
    )
    print(analise.to_string(index=False))
    print("\nModelos selecionados:\n", resumo.to_string(index=False))


if __name__ == "__main__":
    main()
