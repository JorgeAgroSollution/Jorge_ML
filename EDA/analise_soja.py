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
# produção por ano

plt.figure(figsize=(10, 6))

sns.lineplot(data=df, x="ano", y="producao_t", marker="o", color="blue")

plt.title("Produção de Soja por Ano")
plt.xlabel("Ano")
plt.ylabel("Produção (t)")
plt.grid()

# %% [markdown]
# Há uma clara tendencia de crescimento da produção ao longo do tempo. Essa é uma tendencia natural devido ao desenvolvimento tecnologico e aquecimento do clima

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

# %%
