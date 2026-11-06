"""API HTTP para predição dos modelos agrícolas persistidos.

Execute com:
    uv run uvicorn api:app --reload
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

import clima
import config as C


class ClimaDiario(BaseModel):
    """Observação diária compatível com as entradas normalizadas do módulo clima."""

    model_config = ConfigDict(extra="forbid")

    data: str
    temp_media: float
    temp_max: float
    temp_min: float
    precip_mm: float = Field(ge=0)
    umidade_rel: float = Field(ge=0, le=100)
    radiacao_mj: float = Field(ge=0)
    vento_ms: float = Field(ge=0)
    ps_kpa: float | None = Field(default=None, gt=0)
    umidade_solo_raiz: float


class SoloEntrada(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ph_h2o: float
    carbono_organico: float
    ctc: float
    saturacao_bases: float
    aluminio: float
    argila: float
    silte: float
    prof_max_cm: float = Field(ge=0)


class MunicipioEntrada(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cod_ibge: int = Field(gt=0)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class PredicaoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cultura: str = Field(min_length=1)
    ano: int = Field(ge=C.ANO_INICIAL, le=C.ANO_FINAL + 10)
    municipio: MunicipioEntrada
    solo: SoloEntrada
    clima_diario: list[ClimaDiario] = Field(min_length=1)
    lag1_rel: float

    @model_validator(mode="after")
    def cultura_conhecida(self) -> "PredicaoRequest":
        if self.cultura not in C.CALENDARIOS:
            raise ValueError(f"Cultura não suportada: {self.cultura}")
        return self


class PredicaoResponse(BaseModel):
    cultura: str
    cod_ibge: int
    ano: int
    rendimento_predito_kg_ha: float
    modelo: str
    features_utilizadas: list[str]


class ModeloInfo(BaseModel):
    cultura: str
    modelo: str
    features: list[str]
    artefato: str
    unidade_predicao: str


class HealthResponse(BaseModel):
    status: str
    modelos_disponiveis: int


def _diretorio_modelos() -> Path:
    return Path(os.getenv("AGRO_ML_MODELOS_DIR", str(C.DIR_RESULTADOS / "modelos_cultura")))


def _artefato_soja() -> Path:
    return Path(
        os.getenv("AGRO_ML_MODELO_SOJA", str(C.DIR_RESULTADOS / "modelo_soja_final.joblib"))
    )


def _caminhos_modelos() -> dict[str, Path]:
    caminhos = {
        p.stem.removeprefix("modelo_"): p
        for p in _diretorio_modelos().glob("modelo_*.joblib")
    }
    soja = _artefato_soja()
    if soja.exists():
        caminhos["soja"] = soja
    return caminhos


@lru_cache(maxsize=None)
def _carregar(cultura: str) -> dict:
    caminho = _caminhos_modelos().get(cultura)
    if caminho is None:
        raise FileNotFoundError(f"Modelo não encontrado para a cultura '{cultura}'.")
    try:
        artefato = joblib.load(caminho)
    except Exception as exc:
        raise RuntimeError(f"Não foi possível carregar o artefato {caminho}.") from exc
    modelo = artefato.get("modelo")
    features = artefato.get("features") or artefato.get("conjunto_features")
    if modelo is None or not features:
        raise ValueError(f"Artefato inválido para '{cultura}': modelo/features ausentes.")
    return {
        "modelo": modelo,
        "features": list(features),
        "modelo_nome": artefato.get("modelo_nome", type(modelo).__name__),
        "caminho": caminho,
        "unidade_predicao": artefato.get("unidade_predicao", "kg/ha"),
    }


def catalogo_modelos() -> list[ModeloInfo]:
    resultado = []
    for cultura, caminho in sorted(_caminhos_modelos().items()):
        try:
            artefato = _carregar(cultura)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        resultado.append(
            ModeloInfo(
                cultura=cultura,
                modelo=artefato["modelo_nome"],
                features=artefato["features"],
                artefato=str(caminho),
                unidade_predicao=artefato["unidade_predicao"],
            )
        )
    return resultado


def _features_brutas(req: PredicaoRequest, features: list[str]) -> pd.DataFrame:
    diario = pd.DataFrame([item.model_dump() for item in req.clima_diario])
    diario["umidade_solo_raiz"] = pd.to_numeric(
        diario["umidade_solo_raiz"], errors="coerce"
    )
    diario = clima.preparar_diario(diario)
    altitude = clima.altitude_da_celula(diario)
    solo = req.solo.model_dump()
    cad = clima.cad_mm(solo.get("argila"), solo.get("silte"), solo.get("prof_max_cm"))
    features_clima = clima.features_cultura(
        diario,
        req.cultura,
        range(req.ano, req.ano + 1),
        req.municipio.latitude,
        altitude,
        cad,
    )
    linha = features_clima.iloc[0].to_dict()
    linha.update(solo)
    linha["altitude_m"] = altitude
    linha["cad_mm"] = cad
    linha["lag1_rel"] = req.lag1_rel
    ausentes = [f for f in features if f not in linha]
    if ausentes:
        raise ValueError(f"Features não produzidas pela entrada bruta: {ausentes}")
    return pd.DataFrame([{f: linha.get(f) for f in features}])


def prever(req: PredicaoRequest) -> PredicaoResponse:
    try:
        artefato = _carregar(req.cultura)
        entrada = _features_brutas(req, artefato["features"])
        pred_log = float(artefato["modelo"].predict(entrada)[0])
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Falha inesperada durante a predição.") from exc
    rendimento = float(np.exp(pred_log))
    if not np.isfinite(rendimento) or rendimento < 0:
        raise HTTPException(status_code=500, detail="O modelo produziu uma predição inválida.")
    return PredicaoResponse(
        cultura=req.cultura,
        cod_ibge=req.municipio.cod_ibge,
        ano=req.ano,
        rendimento_predito_kg_ha=rendimento,
        modelo=artefato["modelo_nome"],
        features_utilizadas=artefato["features"],
    )


app = FastAPI(
    title="Agro ML Prediction API",
    version="0.1.0",
    description="Predição de rendimento agrícola em kg/ha a partir de clima diário e solo.",
)


@app.get("/health", response_model=HealthResponse, tags=["sistema"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", modelos_disponiveis=len(_caminhos_modelos()))


@app.get("/modelos", response_model=list[ModeloInfo], tags=["modelos"])
def modelos() -> list[ModeloInfo]:
    return catalogo_modelos()


@app.post("/predicoes", response_model=PredicaoResponse, tags=["predições"])
def predicoes(req: PredicaoRequest) -> PredicaoResponse:
    return prever(req)


__all__ = ["app", "catalogo_modelos", "prever"]
