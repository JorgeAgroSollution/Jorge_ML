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
import predicao


class ClimaDiario(BaseModel):
    """Observação diária compatível com as entradas normalizadas do módulo clima."""

    model_config = ConfigDict(extra="forbid")

    data: str = Field(description="Data da observação no formato ISO 8601 (AAAA-MM-DD).")
    temp_media: float = Field(description="Temperatura média diária, em °C.")
    temp_max: float = Field(description="Temperatura máxima diária, em °C.")
    temp_min: float = Field(description="Temperatura mínima diária, em °C.")
    precip_mm: float = Field(
        ge=0, description="Precipitação acumulada no dia, em milímetros."
    )
    umidade_rel: float = Field(
        ge=0, le=100, description="Umidade relativa média do ar, em porcentagem."
    )
    radiacao_mj: float = Field(
        ge=0, description="Radiação solar diária, em MJ/m²/dia."
    )
    vento_ms: float = Field(ge=0, description="Velocidade média do vento, em m/s.")
    ps_kpa: float | None = Field(
        default=None,
        gt=0,
        description="Pressão atmosférica média, em kPa. Opcional; pode ser estimada.",
    )
    umidade_solo_raiz: float = Field(
        description="Umidade relativa do solo na zona radicular, conforme a fonte climática."
    )


class SoloEntrada(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ph_h2o: float = Field(description="pH medido em água.")
    carbono_organico: float = Field(description="Teor de carbono orgânico do solo.")
    ctc: float = Field(description="Capacidade de troca catiônica.")
    saturacao_bases: float = Field(description="Saturação por bases, em porcentagem.")
    aluminio: float = Field(description="Teor de alumínio do solo.")
    argila: float = Field(description="Teor de argila, em g/kg.")
    silte: float = Field(description="Teor de silte, em g/kg.")
    prof_max_cm: float = Field(
        ge=0, description="Profundidade máxima representada pelo perfil, em cm."
    )


class MunicipioEntrada(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cod_ibge: int = Field(gt=0, description="Código IBGE do município.")
    latitude: float = Field(
        ge=-90, le=90, description="Latitude representativa do município, em graus."
    )
    longitude: float = Field(
        ge=-180, le=180, description="Longitude representativa do município, em graus."
    )


class PredicaoRequest(BaseModel):
    """Entrada completa: dados locais de município, solo e clima diário."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "cultura": "milho",
                "ano": 2026,
                "municipio": {
                    "cod_ibge": 3500105,
                    "latitude": -22.3,
                    "longitude": -48.3,
                },
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
                "lag1_rel": 0.0,
                "clima_diario": [
                    {
                        "data": "2025-10-01",
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
                ],
            }
        },
    )

    cultura: str = Field(
        min_length=1,
        description="Cultura modelada. Deve corresponder a uma cultura disponível no catálogo.",
        examples=["milho"],
    )
    ano: int = Field(
        ge=C.ANO_INICIAL,
        le=C.ANO_FINAL + 10,
        description="Ano da safra a ser prevista.",
        examples=[2026],
    )
    municipio: MunicipioEntrada = Field(
        description="Identificação e coordenadas do município da previsão."
    )
    solo: SoloEntrada = Field(
        description="Perfil de solo usado para calcular as variáveis edáficas."
    )
    clima_diario: list[ClimaDiario] = Field(
        min_length=1,
        description=(
            "Série diária usada para calcular as features fenológicas. "
            "Envie todos os dias necessários às janelas da cultura, sem lacunas."
        ),
    )
    lag1_rel: float = Field(
        description="Variação relativa do rendimento da safra anterior no município."
    )

    @model_validator(mode="after")
    def cultura_conhecida(self) -> "PredicaoRequest":
        if self.cultura not in C.CALENDARIOS:
            raise ValueError(f"Cultura não suportada: {self.cultura}")
        return self


class SoloOpcional(BaseModel):
    """Solo do talhão. Todos os campos são opcionais: o que faltar vem do perfil do município."""

    model_config = ConfigDict(extra="forbid")

    ph_h2o: float | None = Field(default=None, description="pH medido em água.")
    carbono_organico: float | None = Field(default=None, description="Teor de carbono orgânico.")
    ctc: float | None = Field(default=None, description="Capacidade de troca catiônica.")
    saturacao_bases: float | None = Field(
        default=None, description="Saturação por bases, em porcentagem."
    )
    aluminio: float | None = Field(default=None, description="Teor de alumínio do solo.")
    argila: float | None = Field(default=None, description="Teor de argila, em g/kg.")
    silte: float | None = Field(default=None, description="Teor de silte, em g/kg.")
    prof_max_cm: float | None = Field(
        default=None, ge=0, description="Profundidade máxima representada, em cm."
    )


class PredicaoSimplesRequest(BaseModel):
    """Mínimo que o agricultor precisa informar: cultura, ano da safra e localização."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "cultura": "milho",
                "ano": 2026,
                "latitude": -21.17,
                "longitude": -47.81,
                "solo": {"ph_h2o": 5.5, "argila": 350, "prof_max_cm": 100},
                "lag1_rel": 0.02,
            }
        },
    )

    cultura: str = Field(
        min_length=1,
        description="Cultura modelada. Deve corresponder a uma cultura disponível no catálogo.",
        examples=["milho"],
    )
    ano: int = Field(
        ge=C.ANO_INICIAL,
        le=C.ANO_FINAL + 10,
        description="Ano da safra a ser prevista.",
        examples=[2026],
    )
    latitude: float = Field(
        ge=-90, le=90, description="Latitude do local, em graus.", examples=[-21.17]
    )
    longitude: float = Field(
        ge=-180,
        le=180,
        description="Longitude do local, em graus.",
        examples=[-47.81],
    )
    solo: SoloOpcional | None = Field(
        default=None,
        description="Dados do talhão. Campos informados substituem o perfil municipal.",
    )
    lag1_rel: float | None = Field(
        default=None,
        description=(
            "Variação relativa do rendimento anterior. Se omitida, usa o histórico "
            "do município mais próximo disponível."
        ),
    )

    @model_validator(mode="after")
    def cultura_conhecida(self) -> "PredicaoSimplesRequest":
        if self.cultura not in C.CALENDARIOS:
            raise ValueError(f"Cultura não suportada: {self.cultura}")
        return self


class PredicaoSimplesResponse(BaseModel):
    """Resultado com estimativa, intervalo e indicação das fontes usadas."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "cultura": "milho",
            "ano": 2026,
            "rendimento_predito_kg_ha": 5840.2,
            "p10_kg_ha": 4310.0,
            "p90_kg_ha": 7365.4,
            "referencia_municipio_kg_ha": 5602.1,
            "municipio_cod_ibge": 3543402,
            "distancia_km": 0.0,
            "solo_origem": "perfil_local",
            "lag_origem": "perfil_local",
            "clima_usado": "clima_tipico",
            "confiavel": True,
            "avisos": [],
            "modelo": "HistGradientBoostingRegressor",
        }
    })

    cultura: str
    ano: int
    rendimento_predito_kg_ha: float
    p10_kg_ha: float
    p90_kg_ha: float
    referencia_municipio_kg_ha: float
    municipio_cod_ibge: int
    distancia_km: float
    solo_origem: str
    lag_origem: str
    clima_usado: str
    confiavel: bool
    avisos: list[str]
    modelo: str


class PredicaoResponse(BaseModel):
    """Resultado da predição completa e lista das features fornecidas ao modelo."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "cultura": "milho",
            "cod_ibge": 3500105,
            "ano": 2026,
            "rendimento_predito_kg_ha": 5840.2,
            "modelo": "HistGradientBoostingRegressor",
            "features_utilizadas": ["chuva_floracao_mm", "cad_mm", "lag1_rel"],
        }
    })

    cultura: str = Field(description="Cultura prevista.")
    cod_ibge: int = Field(description="Código IBGE do município.")
    ano: int = Field(description="Ano da safra.")
    rendimento_predito_kg_ha: float = Field(
        description="Rendimento estimado, em quilogramas por hectare."
    )
    modelo: str = Field(description="Nome do estimador usado na predição.")
    features_utilizadas: list[str] = Field(
        description="Nomes das features efetivamente entregues ao modelo."
    )


class ModeloInfo(BaseModel):
    """Metadados de um modelo disponível para uma cultura."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "cultura": "milho",
            "modelo": "HistGradientBoostingRegressor",
            "features": ["chuva_floracao_mm", "cad_mm", "lag1_rel"],
            "artefato": "resultados/modelos_cultura/modelo_milho.joblib",
            "unidade_predicao": "kg/ha",
        }
    })

    cultura: str = Field(description="Cultura atendida pelo modelo.")
    modelo: str = Field(description="Nome do estimador persistido.")
    features: list[str] = Field(description="Features esperadas pelo estimador.")
    artefato: str = Field(description="Caminho do artefato carregado no servidor.")
    unidade_predicao: str = Field(description="Unidade do valor previsto.")


class HealthResponse(BaseModel):
    """Estado da API e quantidade de modelos carregáveis."""

    model_config = ConfigDict(json_schema_extra={
        "example": {"status": "ok", "modelos_disponiveis": 9}
    })

    status: str = Field(description="Estado da aplicação; atualmente, 'ok' indica disponibilidade.")
    modelos_disponiveis: int = Field(
        description="Quantidade de culturas com artefatos de modelo disponíveis."
    )


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
        caminhos.setdefault("soja", soja)  # o artefato legado só vale se não houver um novo na pasta
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
        "bruto": artefato,
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
    rendimento = predicao.rendimento_kg(artefato["bruto"], req.ano, pred_log)
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
    description=(
        "API para estimar o rendimento agrícola em kg/ha. "
        "Use /predicoes/simples quando não possuir clima diário; use /predicoes "
        "para enviar dados climáticos e de solo completos. A API não coleta dados "
        "externos durante a requisição."
    ),
    contact={"name": "Equipe Agro ML"},
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["sistema"],
    summary="Verificar disponibilidade da API",
    description="Retorna o estado da aplicação e quantos modelos estão disponíveis para predição.",
)
def health() -> HealthResponse:
    return HealthResponse(status="ok", modelos_disponiveis=len(_caminhos_modelos()))


@app.get(
    "/modelos",
    response_model=list[ModeloInfo],
    tags=["modelos"],
    summary="Listar modelos disponíveis",
    description=(
        "Lista as culturas que podem ser previstas e os metadados dos artefatos "
        "carregados. Use este endpoint antes de enviar uma cultura para /predicoes."
    ),
)
def modelos() -> list[ModeloInfo]:
    return catalogo_modelos()


def prever_simples(req: PredicaoSimplesRequest) -> PredicaoSimplesResponse:
    try:
        artefato = _carregar(req.cultura)
        solo = req.solo.model_dump() if req.solo else None
        saida = predicao.prever_simples(
            artefato["bruto"], req.ano, req.latitude, req.longitude, solo, req.lag1_rel
        )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Falha inesperada durante a predição.") from exc
    return PredicaoSimplesResponse(
        cultura=req.cultura, ano=req.ano, modelo=artefato["modelo_nome"], **saida
    )


@app.post(
    "/predicoes/simples",
    response_model=PredicaoSimplesResponse,
    tags=["predições"],
    summary="Prever rendimento com dados mínimos",
    description=(
        "Estima o rendimento usando cultura, ano e coordenadas. O serviço busca "
        "o município e o perfil de solo mais apropriados no artefato do modelo, "
        "e usa clima típico quando não há clima futuro informado. O solo e o "
        "rendimento relativo do ano anterior são opcionais."
    ),
)
def predicoes_simples(req: PredicaoSimplesRequest) -> PredicaoSimplesResponse:
    return prever_simples(req)


@app.post(
    "/predicoes",
    response_model=PredicaoResponse,
    tags=["predições"],
    summary="Prever rendimento com dados climáticos e de solo",
    description=(
        "Calcula as features fenológicas a partir do clima diário, combina-as "
        "com o perfil de solo e aplica o modelo da cultura. Envie uma série "
        "diária com cobertura suficiente para todas as janelas fenológicas da "
        "cultura; uma lista curta ou com lacunas pode gerar erro 422."
    ),
)
def predicoes(req: PredicaoRequest) -> PredicaoResponse:
    return prever(req)


__all__ = ["app", "catalogo_modelos", "prever", "prever_simples"]
