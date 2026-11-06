# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: agro-ml (3.13.14)
#     language: python
#     name: python3
# ---

# %%
# imports
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde
import joblib

# Garante que a raiz do projeto esteja disponível para importação
raiz_projeto = Path.cwd().parent
if str(raiz_projeto) not in sys.path:
    sys.path.insert(0, str(raiz_projeto))

import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import catalogo_variaveis
import unidades
import config

# %% [markdown]
# # Carga dos dados

# %%
df = pd.read_csv("../data/processed/dataset_soja.csv")

# %%
df.shape

# %%
df.info()

# %%
df.describe()

# %%
df.describe(include="object")

# %%
df.loc[df["ano"] >= 2022]

# %%
df = df[df["solo_imputado"].eq(0)]


# %% [markdown]
# # Análise das variáveis numéricas

# %% [markdown]
# ## Qualidade dos dados

# %%
def plot_missing_values(df, figsize=(12, 6)):
    missing_pct = df.isnull().mean().mul(100).sort_values(ascending=False)

    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        print("Nenhuma feature possui valores ausentes.")
        return

    plt.figure(figsize=figsize)

    ax = sns.barplot(x=missing_pct.values, y=missing_pct.index, color="steelblue")

    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)

    plt.title("Porcentagem de Valores Ausentes por Feature")
    plt.xlabel("Valores ausentes (%)")
    plt.ylabel("Features")
    plt.xlim(0, min(100, missing_pct.max() * 1.15))
    plt.tight_layout()
    plt.show()


# %%
plot_missing_values(df)

# %%
# Dropando colunas com mais de 30% de valores ausentes
df.drop(columns=df.columns[df.isnull().mean() > 0.30], inplace=True)


# %%
def analyze_missing_rows(df, threshold=0.30):
    missing_count = df.isnull().sum(axis=1)
    missing_pct = df.isnull().mean(axis=1) * 100

    result = pd.DataFrame(
        {
            "linha": df.index,
            "features_ausentes": missing_count.values,
            "features_ausentes_pct": missing_pct.values,
        }
    )

    result = (
        result[result["features_ausentes_pct"] > threshold * 100]
        .sort_values("features_ausentes_pct", ascending=False)
        .reset_index(drop=True)
    )

    print(f"Total de linhas: {len(df)}")
    print(f"Linhas com mais de {threshold:.0%} de features ausentes: {len(result)}")
    print(f"Percentual das linhas: {len(result) / len(df) * 100:.2f}%")

    return result


# %%
df_missing_rows = analyze_missing_rows(df, threshold=0.10)

display(df_missing_rows)

# %%
# Dropping enchimento_dias_geada_d
df.drop(columns=["enchimento_dias_geada"], inplace=True)

# %% [markdown]
# ## Univariada

# %%
columns_to_plot = [
    "ano",
    "producao_t",
    "rendimento_kg_ha",
    "area_colhida_ha",
    "area_plantada_ha",
    "peso",
]

columns_to_plot_implant = [
    "implant_prec",
    "implant_tmax",
    "implant_tmin",
    "implant_rad",
    "implant_defic",
    "implant_estresse",
    "implant_dias_calor",
    "implant_dias_geada",
    "implant_seca_max",
    "implant_gd",
    "implant_umsolo",
]


# %%
def plot_histograms_kde(df, columns, bins=30, ncols=2):
    nrows = math.ceil(len(columns) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        data = df[col].dropna()

        if data.empty:
            ax.set_title(f"Sem dados: {col}")
            continue

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        limite_superior = q3 + 3.5 * iqr

        x_min = data.min()
        data_filtrada = data[data <= limite_superior]
        bins_range = np.linspace(x_min, limite_superior, bins + 1)

        ax.hist(
            data_filtrada,
            bins=bins_range,
            density=True,
            color="skyblue",
            edgecolor="black",
            alpha=0.6,
            label="Histograma",
        )

        if len(data_filtrada) > 1 and data_filtrada.nunique() > 1:
            kde = gaussian_kde(data_filtrada)
            x = np.linspace(x_min, limite_superior, 300)
            ax.plot(x, kde(x), color="red", linewidth=2, label="KDE")

        media = data_filtrada.mean()
        mediana = data_filtrada.median()

        ax.axvline(
            media,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Média: {media:.2f}",
        )
        ax.axvline(
            mediana,
            color="green",
            linestyle="-.",
            linewidth=2,
            label=f"Mediana: {mediana:.2f}",
        )

        ax.set_xlim(x_min, limite_superior)
        ax.set_title(f"Distribuição de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Densidade")
        ax.legend()

    for ax in axes[len(columns) :]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


# %%
original_columns = df.columns.tolist()
df = unidades.para_exportacao(df)

# %%
df.describe()

# %%
plot_histograms_kde(df, columns_to_plot, bins=50, ncols=2)

# %%
plot_histograms_kde(
    df, [x for x in catalogo_variaveis.SOLO if x in df.columns], bins=50, ncols=2
)

# %%
columns_to_plot_clim = [
    f"enchimento_{i}"
    for i in catalogo_variaveis.CLIMA_INDICADORES
    if f"enchimento_{i}" in df.columns
]

# %%
plot_histograms_kde(df, columns_to_plot_clim, bins=50, ncols=2)


# %%
def plot_boxplots(df, columns, ncols=2, showfliers=True):
    nrows = math.ceil(len(columns) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        data = df[col].dropna()

        if data.empty:
            ax.set_title(f"Sem dados: {col}")
            continue

        ax.boxplot(
            data,
            vert=True,
            patch_artist=True,
            showfliers=showfliers,
            boxprops=dict(facecolor="skyblue", edgecolor="black"),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(
                marker="o",
                markerfacecolor="orange",
                markeredgecolor="black",
                markersize=4,
                alpha=0.6,
            ),
        )

        ax.set_title(f"Boxplot de {col}")
        ax.set_ylabel(col)
        ax.set_xticks([])

    for ax in axes[len(columns) :]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


# %%
plot_boxplots(df, columns_to_plot_clim, ncols=2)

# %%
# Dropando colunas com menos de 80% de preenchimento
df_eda = df.copy()
df_eda.dropna(axis=1, thresh=int(0.8 * len(df_eda)), inplace=True)

# %% [markdown]
# # Análise multivariada

# %%
# Verificar produção por ano

plt.figure(figsize=(10, 6))

sns.lineplot(data=df, x="ano", y="rendimento_kg_ha", marker="o", color="blue")

plt.title("Produção de Soja por Ano")
plt.xlabel("Ano")
plt.ylabel("Produção (t)")
plt.grid()

# %% [markdown]
# Há uma clara tendencia de crescimento do rendimento ao longo do tempo. Essa é uma tendencia natural devido ao desenvolvimento tecnologico e aquecimento do clima

# %% [markdown]
# # Todo: Verificar as métricas entre o vale de 2014 e o pico de 2024

# %%
# relação area plantada e colhida


def plot_scatter_subplots(df, plots, ncols=2):
    nrows = math.ceil(len(plots) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10 * ncols, 6 * nrows), squeeze=False
    )
    axes = axes.ravel()

    for ax, (x, y, title, xlabel, ylabel) in zip(axes, plots):
        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            marker="o",
            hue="ano",
            palette="viridis",
            ax=ax,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid()

    for ax in axes[len(plots) :]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


plot_scatter_subplots(
    df,
    [
        (
            "rendimento_kg_ha",
            "area_plantada_ha",
            "Rendimento e Área Plantada",
            "Rendimento (kg/ha)",
            "Área plantada (ha)",
        ),
        (
            "rendimento_kg_ha",
            "implant_tmax_c",
            "Rendimento e Área Colhida",
            "Rendimento (kg/ha)",
            "Área colhida (ha)",
        ),
        (
            "rendimento_kg_ha",
            "implant_tmin_c",
            "Rendimento e Área Colhida",
            "Rendimento (kg/ha)",
            "Área colhida (ha)",
        ),
    ],
)


# %% [markdown]
# Não há uma discriminação clara associada à relação entre a area plantada e o rendimento. Surge a hipótese que há outros fatores que influenciam um melhor rendimento. 

# %%
def plot_correlation_by_year(df, columns, year_col="ano", year=None):
    if year is not None:
        data = df.loc[df[year_col] == year, columns]
    else:
        data = df[columns]

    plt.figure(figsize=(10, 10))

    sns.heatmap(
        data.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
    )

    title = (
        f"Correlação entre features — {year}"
        if year is not None
        else "Correlação entre features"
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


# %%
columns = [
    "producao_t",
    "rendimento_kg_ha",
    "area_colhida_ha",
    "area_plantada_ha",
    "peso",
]

plot_correlation_by_year(df, columns, year_col="ano", year=2025)

# %% [markdown]
# # Modelo

# %%
from sklearn.model_selection import train_test_split


# Definição do target e features
df.columns = original_columns
X = df.columns.difference(config.COLS_VAZAMENTO)
Y = "rendimento"

# %%
df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=42)

# %% [markdown]
# ## Modelagem reproduzível
#
# O `train_test_split` acima é mantido como referência da EDA, mas não será
# usado para reportar desempenho. Como os dados formam um painel municipal,
# os experimentos abaixo separam o teste antes do tuning e avaliam dois
# cenários: novas observações dos mesmos municípios e municípios não vistos.
# `ano`, identificadores e variáveis derivadas do alvo ficam fora de `X`.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def preparar_modelagem(df):
    """Normaliza o CSV exportado e retorna dados prontos para os experimentos."""
    dados = unidades.para_modelagem(df.copy())
    alvo_col = "rendimento"
    if alvo_col not in dados:
        raise KeyError("O dataset precisa conter rendimento_kg_ha ou rendimento.")
    dados = dados.dropna(subset=[alvo_col]).copy()
    dados = dados[dados[alvo_col] > 0].copy()

    colunas_excluir = set(config.COLS_VAZAMENTO) | set(config.COLS_REDUNDANTES)
    colunas_excluir.update({"latitude", "longitude", "lat", "lon"})
    features = [
        c
        for c in dados.select_dtypes(include="number").columns
        if c not in colunas_excluir
        and c != alvo_col
        and dados[c].notna().mean() > 0.5
        and dados[c].nunique(dropna=True) > 1
    ]
    if not features:
        raise ValueError("Nenhuma feature numérica válida foi encontrada.")
    if "cod_ibge" not in dados:
        raise KeyError("cod_ibge é necessário para os cenários agrupados.")

    dados["y_log_modelo"] = np.log(dados[alvo_col])
    dados["bin_alvo"] = pd.qcut(
        dados["y_log_modelo"], q=10, labels=False, duplicates="drop"
    )
    return dados, features


dados_modelo, features_modelo = preparar_modelagem(
    pd.read_csv("../data/processed/dataset_soja.csv")
)
print(
    f"{len(dados_modelo):,} linhas, {dados_modelo['cod_ibge'].nunique():,} municípios, "
    f"{len(features_modelo)} features"
)

# %%
# O teste é um holdout de municípios inteiros. Ele fica intocado até a célula
# final, que apenas prepara as previsões para execução explícita pelo usuário.
rng = np.random.default_rng(42)
municipios = dados_modelo["cod_ibge"].drop_duplicates().to_numpy()
municipios_teste = set(
    rng.choice(municipios, size=max(1, round(0.20 * len(municipios))), replace=False)
)
eh_teste = dados_modelo["cod_ibge"].isin(municipios_teste)
desenvolvimento = dados_modelo.loc[~eh_teste].reset_index(drop=True)
teste_soja = dados_modelo.loc[eh_teste].reset_index(drop=True)

X_dev = desenvolvimento[features_modelo]
y_dev = desenvolvimento["y_log_modelo"]
grupos_dev = desenvolvimento["cod_ibge"]
bins_dev = desenvolvimento["bin_alvo"]
print(
    f"desenvolvimento={len(desenvolvimento):,} linhas / "
    f"{desenvolvimento['cod_ibge'].nunique():,} municípios; "
    f"teste reservado={len(teste_soja):,} linhas / "
    f"{teste_soja['cod_ibge'].nunique():,} municípios"
)

# %%
def pipeline_modelo(nome):
    """Tratamentos aprendidos no fold, seguidos de um regressor."""
    base = [("imputacao", SimpleImputer(strategy="median"))]
    if nome == "ElasticNet":
        base.append(("escala", StandardScaler()))
        base.append(("modelo", ElasticNet(max_iter=10000, random_state=42)))
    elif nome == "RandomForest":
        base.append(
            (
                "modelo",
                RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1),
            )
        )
    elif nome == "HistGradientBoosting":
        base.append(
            (
                "modelo",
                HistGradientBoostingRegressor(max_iter=300, random_state=42),
            )
        )
    else:
        raise ValueError(f"Modelo desconhecido: {nome}")
    return Pipeline(base)


parametros = {
    "ElasticNet": {
        "modelo__alpha": np.logspace(-4, 1, 20),
        "modelo__l1_ratio": np.linspace(0.05, 0.95, 10),
    },
    "RandomForest": {
        "modelo__n_estimators": [200, 400, 700],
        "modelo__max_depth": [None, 8, 16, 24],
        "modelo__min_samples_leaf": [1, 2, 5, 10],
        "modelo__max_features": [0.5, 0.8, 1.0],
    },
    "HistGradientBoosting": {
        "modelo__learning_rate": [0.02, 0.05, 0.1],
        "modelo__max_iter": [150, 300, 500],
        "modelo__max_leaf_nodes": [7, 15, 31],
        "modelo__l2_regularization": [0.0, 1.0, 5.0],
        "modelo__min_samples_leaf": [10, 25, 50],
    },
}


def smape(y_real, y_previsto):
    denominador = np.abs(y_real) + np.abs(y_previsto)
    return float(
        np.mean(2 * np.abs(y_real - y_previsto) / np.maximum(denominador, 1e-9))
    )


def metricas_rendimento(y_log_real, y_log_previsto):
    real = np.exp(np.asarray(y_log_real))
    previsto = np.exp(np.asarray(y_log_previsto))
    return {
        "MAE_kg_ha": mean_absolute_error(real, previsto),
        "RMSE_kg_ha": mean_squared_error(real, previsto) ** 0.5,
        "R2_log": r2_score(y_log_real, y_log_previsto),
        "sMAPE": smape(real, previsto),
    }


def ajustar_modelos(dados, features, modo="mesmos_municipios", n_splits=5):
    """Tuning interno + validação externa, sem acessar o holdout."""
    X = dados[features]
    y = dados["y_log_modelo"]
    grupos = dados["cod_ibge"]
    bins = dados["bin_alvo"]
    if modo == "mesmos_municipios":
        cv_externo = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        # A busca recebe o alvo contínuo; estratificação por bins já é feita
        # no loop externo. No loop interno, KFold evita tratar cada valor
        # contínuo como uma classe distinta.
        cv_interno = KFold(n_splits=4, shuffle=True, random_state=43)
        split_externo = lambda: cv_externo.split(X, bins)
        split_interno = cv_interno
    elif modo == "municipios_novos":
        cv_externo = GroupKFold(n_splits=n_splits)
        cv_interno = GroupKFold(n_splits=4)
        split_externo = lambda: cv_externo.split(X, y, grupos)
        split_interno = cv_interno
    else:
        raise ValueError("modo deve ser 'mesmos_municipios' ou 'municipios_novos'")

    linhas = []
    melhores = {}
    for nome in parametros:
        folds = []
        for fold, (idx_treino, idx_validacao) in enumerate(split_externo(), 1):
            busca = RandomizedSearchCV(
                pipeline_modelo(nome),
                parametros[nome],
                n_iter=6,
                scoring="neg_root_mean_squared_error",
                cv=split_interno,
                random_state=42 + fold,
                n_jobs=1,
                refit=True,
            )
            grupos_internos = (
                grupos.iloc[idx_treino] if modo == "municipios_novos" else None
            )
            busca.fit(
                X.iloc[idx_treino],
                y.iloc[idx_treino],
                **({"groups": grupos_internos} if grupos_internos is not None else {}),
            )
            pred = busca.predict(X.iloc[idx_validacao])
            m = metricas_rendimento(y.iloc[idx_validacao], pred)
            m.update(modelo=nome, modo=modo, fold=fold)
            folds.append(m)
            melhores[(nome, modo, fold)] = busca.best_estimator_
        linhas.extend(folds)
    return pd.DataFrame(linhas), melhores


# %%
resultados_mesmos, modelos_mesmos = ajustar_modelos(
    desenvolvimento, features_modelo, modo="mesmos_municipios"
)
resultados_novos, modelos_novos = ajustar_modelos(
    desenvolvimento, features_modelo, modo="municipios_novos"
)
resultados_validacao = pd.concat(
    [resultados_mesmos, resultados_novos], ignore_index=True
)
resumo_validacao = (
    resultados_validacao.groupby(["modo", "modelo"])[
        ["MAE_kg_ha", "RMSE_kg_ha", "R2_log", "sMAPE"]
    ]
    .agg(["mean", "std"])
    .sort_values(("MAE_kg_ha", "mean"))
)
display(resumo_validacao.round(3))

# %%
# Baseline: mediana do rendimento no treino de cada fold.
baseline = []
for modo, dados_cv in [
    ("mesmos_municipios", resultados_mesmos),
    ("municipios_novos", resultados_novos),
]:
    mediana_log = desenvolvimento["y_log_modelo"].median()
    baseline.append(
        {
            "modo": modo,
            "modelo": "Baseline_mediana",
            "MAE_kg_ha": metricas_rendimento(
                desenvolvimento["y_log_modelo"],
                np.full(len(desenvolvimento), mediana_log),
            )["MAE_kg_ha"],
        }
    )
display(pd.DataFrame(baseline))

# %%
# Diagnóstico agregado para orientar o segundo loop: erros por ano e por região.
# A região é a célula climática aproximada quando lat/lon está disponível.
def diagnostico_residuos(dados, modelo_por_fold, modo):
    linhas = []
    if "lat" in dados and "lon" in dados:
        regiao = (
            dados["lat"].round(1).astype(str) + "_" + dados["lon"].round(1).astype(str)
        )
    else:
        regiao = dados["cod_ibge"].astype(str)
    if modo == "mesmos_municipios":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = cv.split(dados[features_modelo], dados["bin_alvo"])
    else:
        cv = GroupKFold(n_splits=5)
        splits = cv.split(dados[features_modelo], groups=dados["cod_ibge"])
    for fold, (_, idx_val) in enumerate(splits, 1):
        modelo = modelo_por_fold[("HistGradientBoosting", modo, fold)]
        pred = modelo.predict(dados.iloc[idx_val][features_modelo])
        aux = dados.iloc[idx_val][["ano"]].copy()
        aux["regiao"] = regiao.iloc[idx_val].to_numpy()
        aux["erro_kg_ha"] = np.exp(dados.iloc[idx_val]["y_log_modelo"]) - np.exp(pred)
        linhas.append(aux)
    return pd.concat(linhas, ignore_index=True)


residuos_mesmos = diagnostico_residuos(
    desenvolvimento, modelos_mesmos, "mesmos_municipios"
)
residuos_novos = diagnostico_residuos(
    desenvolvimento, modelos_novos, "municipios_novos"
)
display(
    residuos_mesmos.groupby("ano")["erro_kg_ha"]
    .agg(["mean", "median", "count"])
    .round(1)
)
display(
    residuos_novos.groupby("regiao")["erro_kg_ha"]
    .agg(["mean", "median", "count"])
    .sort_values("mean")
    .head(15)
    .round(1)
)

# %% [markdown]
# ## Segundo loop: melhorias guiadas pela validação
#
# O loop compara o conjunto completo de features com uma versão reduzida,
# removendo variáveis redundantes e features com ausência elevada. A escolha
# final deve ser feita apenas pela média da validação e pela estabilidade
# entre folds; o conjunto de teste continua sem ser consultado.

# %%
features_reduzidas = [
    c
    for c in features_modelo
    if c not in {"temp_media", "temp_min_media", "umidade_media", "vento_medio_ms"}
]
experimentos = {"completa": features_modelo, "reduzida": features_reduzidas}
melhorias = []
for nome_features, features in experimentos.items():
    r, _ = ajustar_modelos(
        desenvolvimento, features, modo="mesmos_municipios", n_splits=5
    )
    resumo = r.groupby("modelo")["MAE_kg_ha"].mean().sort_values()
    melhorias.extend(
        {"features": nome_features, "modelo": modelo, "MAE_medio_kg_ha": valor}
        for modelo, valor in resumo.items()
    )
resultado_melhorias = pd.DataFrame(melhorias).sort_values("MAE_medio_kg_ha")
display(resultado_melhorias.round(2))

# %% [markdown]
# ## Treinamento e persistência do modelo final
#
# O modelo final é treinado somente no conjunto de desenvolvimento. O holdout
# municipal não é acessado nesta célula. O artefato inclui o pipeline, a lista
# exata de features e metadados para permitir reprodução posterior.

# %%
melhor_linha = resultado_melhorias.iloc[0]
melhor_nome = melhor_linha["modelo"]
melhor_conjunto_features = melhor_linha["features"]
melhor_features = experimentos[melhor_conjunto_features]
modelo_final = pipeline_modelo(melhor_nome)
modelo_final.set_params(
    **modelos_mesmos[(melhor_nome, "mesmos_municipios", 1)].get_params()
)
modelo_final.fit(desenvolvimento[melhor_features], desenvolvimento["y_log_modelo"])

artefato_modelo = {
    "modelo": modelo_final,
    "features": melhor_features,
    "modelo_nome": melhor_nome,
    "conjunto_features": melhor_conjunto_features,
    "alvo": "log(rendimento_kg_ha)",
    "municipios_treino": desenvolvimento["cod_ibge"].unique(),
    "municipios_teste_reservados": teste_soja["cod_ibge"].unique(),
    "semente": 42,
    "observacao": "Treinado sem acessar o holdout municipal.",
}
config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
caminho_modelo = config.DIR_RESULTADOS / "modelo_soja_final.joblib"
joblib.dump(artefato_modelo, caminho_modelo)
print(f"Modelo salvo em: {caminho_modelo}")

# %% [markdown]
# ## Execução do teste (manual)
#
# Esta célula fica desativada intencionalmente. Depois de revisar e aceitar o
# modelo final, carregue o artefato e execute-a para gerar previsões do holdout.

# %%
artefato_carregado = joblib.load(config.DIR_RESULTADOS / "modelo_soja_final.joblib")
modelo_carregado = artefato_carregado["modelo"]
features_carregadas = artefato_carregado["features"]
previsoes_teste_kg_ha = np.exp(
    modelo_carregado.predict(teste_soja[features_carregadas])
)
saida_teste = teste_soja[["cod_ibge", "ano", "rendimento"]].copy()
saida_teste["previsao_kg_ha"] = previsoes_teste_kg_ha
saida_teste.to_csv(config.DIR_RESULTADOS / "previsoes_teste_soja.csv", index=False)
saida_teste.head()
