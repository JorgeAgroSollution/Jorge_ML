"""Configuração central do pipeline: calendários agrícolas, limites de plausibilidade e quebras de série.

TODAS as premissas agronômicas ficam aqui para serem revisadas por um agrônomo/extensionista.
Os calendários abaixo são uma primeira aproximação para o estado de SP e devem ser calibrados
com o zoneamento (ZARC), com a CATI/IAC e com a realidade do produtor local.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DIR_RAW = RAIZ / "data" / "raw"
DIR_TEMP = RAIZ / "data" / "temp"
DIR_PROC = RAIZ / "data" / "processed"
DIR_CACHE = RAIZ / "data" / "cache"
DIR_RESULTADOS = RAIZ / "resultados"

ANO_INICIAL = 2000
ANO_FINAL = 2025
UF = "SP"

CULTURAS = [
    "soja",
    "milho",
    "feijao",
    "arroz",
    "trigo",
    "algodao",
    "cafe",
    "cana",
    "laranja",
    "mandioca",
]

# --- Quebras de série documentadas na PAM/IBGE (agrobr.ibge.pam) -------------------------------
# laranja <= 2000: rendimento em frutos/ha (mil frutos), não kg/ha.
# café <= 2001: "em coco"; a partir de 2002: "beneficiado". Sem conversão implícita.
PRIMEIRO_ANO_VALIDO = {"laranja": 2001, "cafe": 2002}

# --- Limites de plausibilidade do rendimento (kg/ha) -------------------------------------------
# Definidos com base na distribuição observada (p1-p99) + conhecimento agronômico. Revisar!
LIMITES_RENDIMENTO = {
    "soja": (800, 5500),
    "milho": (800, 12000),
    "feijao": (200, 4000),
    "arroz": (500, 7500),
    "trigo": (500, 5500),
    "algodao": (500, 5500),
    "cafe": (300, 4500),
    "cana": (30000, 160000),
    "laranja": (3000, 90000),
    "mandioca": (3000, 50000),
}

# --- Parâmetros de limpeza ---------------------------------------------------------------------
AREA_COLHIDA_MIN_HA = (
    10  # abaixo disso o rendimento da PAM é basicamente ruído de arredondamento
)
RUN_REPETIDO_MAX = (
    3  # >= 3 anos seguidos com o MESMO rendimento => valor estimado, não medido
)
RAZAO_MUN_MIN, RAZAO_MUN_MAX = 0.25, 4.0  # rendimento / mediana do município-cultura


@dataclass(frozen=True)
class Janela:
    """Janela fenológica. Os offsets são relativos ao ano da PAM (0 = ano da colheita; -1 = ano anterior)."""

    nome: str
    ano0: int
    mes0: int
    ano1: int
    mes1: int


def J(nome: str, a0: int, m0: int, a1: int, m1: int) -> Janela:
    return Janela(nome, a0, m0, a1, m1)


@dataclass(frozen=True)
class Calendario:
    janelas: tuple[Janela, ...]
    kc_mensal: tuple[
        float, ...
    ]  # 12 valores (jan..dez): coeficiente de cultura aproximado
    tbase: float  # temperatura-base para graus-dia (°C)
    t_calor: float  # limiar de estresse por calor (Tmax >= t_calor, °C)
    perene: bool = False
    nota: str = ""


CALENDARIOS: dict[str, Calendario] = {
    "soja": Calendario(
        janelas=(
            J("implant", -1, 10, -1, 11),
            J("floracao", -1, 12, 0, 1),
            J("enchimento", 0, 2, 0, 3),
        ),
        kc_mensal=(1.1, 1.1, 0.8, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.7, 1.0),
        tbase=10,
        t_calor=32,
        nota="Semeadura out-nov, floração dez-jan, enchimento de grãos fev-mar. Veranico em jan-fev é o principal risco.",
    ),
    "milho": Calendario(
        janelas=(
            J("implant", -1, 10, -1, 11),
            J("pendoamento", -1, 12, 0, 1),
            J("enchimento", 0, 2, 0, 3),
        ),
        kc_mensal=(1.15, 1.1, 0.8, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.7, 1.0),
        tbase=10,
        t_calor=33,
        nota="Calendário da 1ª safra. A PAM 'milho' mistura 1ª e 2ª (safrinha); separar as safras se possível.",
    ),
    "feijao": Calendario(
        janelas=(
            J("aguas_implant", -1, 10, -1, 11),
            J("aguas_floracao", -1, 12, 0, 1),
            J("seca_floracao", 0, 3, 0, 5),
        ),
        kc_mensal=(1.0, 0.8, 0.8, 0.9, 0.6, 0.3, 0.3, 0.3, 0.4, 0.5, 0.8, 1.0),
        tbase=10,
        t_calor=32,
        nota="Três safras no ano (águas, seca, inverno); a PAM soma tudo. Calendário simplificado.",
    ),
    "arroz": Calendario(
        janelas=(
            J("implant", -1, 10, -1, 12),
            J("floracao", 0, 1, 0, 2),
            J("enchimento", 0, 3, 0, 4),
        ),
        kc_mensal=(1.1, 1.1, 1.0, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.8, 1.0),
        tbase=11,
        t_calor=35,
        nota="Arroz de sequeiro. Seca na floração causa esterilidade de espiguetas.",
    ),
    "trigo": Calendario(
        janelas=(
            J("implant", 0, 4, 0, 5),
            J("geada_espiga", 0, 6, 0, 8),
            J("enchimento", 0, 9, 0, 10),
        ),
        kc_mensal=(0.3, 0.3, 0.3, 0.5, 0.7, 1.0, 1.1, 1.1, 0.8, 0.5, 0.3, 0.3),
        tbase=5,
        t_calor=30,
        nota="Trigo de sequeiro, semeadura abr-mai. Geada na espigação e chuva na colheita são os riscos.",
    ),
    "algodao": Calendario(
        janelas=(
            J("implant", -1, 11, -1, 12),
            J("florescimento", 0, 1, 0, 2),
            J("capulho", 0, 3, 0, 5),
        ),
        kc_mensal=(1.1, 1.1, 1.0, 0.8, 0.6, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.8),
        tbase=15,
        t_calor=35,
        nota="Semeadura nov-dez.",
    ),
    "cafe": Calendario(
        janelas=(
            J("inverno_ant", -1, 6, -1, 8),
            J("florada", -1, 9, -1, 10),
            J("granacao", -1, 12, 0, 3),
            J("colheita", 0, 5, 0, 8),
        ),
        kc_mensal=(0.95, 0.95, 0.95, 0.9, 0.85, 0.8, 0.8, 0.8, 0.85, 0.9, 0.95, 0.95),
        tbase=10,
        t_calor=32,
        perene=True,
        nota="Perene e bienal: use também a produtividade do ano anterior. Geada no inverno anterior afeta a safra.",
    ),
    "cana": Calendario(
        janelas=(
            J("seca_ant", -1, 5, -1, 9),
            J("brotacao", -1, 10, -1, 12),
            J("crescimento", 0, 1, 0, 3),
            J("maturacao", 0, 4, 0, 9),
        ),
        kc_mensal=(1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.6, 0.6, 0.6, 0.8, 0.95),
        tbase=16,
        t_calor=35,
        perene=True,
        nota="Ciclo de 12-18 meses e corte em estágios diferentes: a produtividade do ano depende do estágio (cana de 1º corte, soca).",
    ),
    "laranja": Calendario(
        janelas=(
            J("pre_florada", -1, 7, -1, 9),
            J("florada_fixacao", -1, 9, -1, 11),
            J("frutificacao", -1, 12, 0, 3),
        ),
        kc_mensal=(0.75, 0.75, 0.75, 0.7, 0.7, 0.65, 0.65, 0.65, 0.65, 0.7, 0.7, 0.75),
        tbase=13,
        t_calor=35,
        perene=True,
        nota="Perene. Déficit hídrico no inverno induz florada; déficit em set-nov reduz a fixação de frutos.",
    ),
    "mandioca": Calendario(
        janelas=(
            J("implant", -1, 10, -1, 12),
            J("crescimento", 0, 1, 0, 4),
            J("repouso_seca", 0, 5, 0, 9),
        ),
        kc_mensal=(0.9, 0.9, 0.9, 0.8, 0.6, 0.4, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8),
        tbase=12,
        t_calor=35,
        perene=True,
        nota="Ciclo de 12-18 meses; geada e seca no repouso afetam o rendimento.",
    ),
}

# Colunas que NUNCA devem entrar como feature (vazamento ou redundância)
COLS_VAZAMENTO = {
    "producao",
    "area_colhida",
    "area_plantada",
    "rendimento",
    "rendimento_calc",
    "razao_colhida_plantada",
    "perda_flag",
    "rend_repetido",
    "y_log",
    "anomalia",
    "peso",
    "ano",
    "produto",
    "localidade",
    "cod_ibge",
    "municipio",
    "n_perfis",
    "solo_dist_km",
    "_mun",
    "solo_imputado",
}
COLS_REDUNDANTES = {
    "latitude_x",
    "latitude_y",
    "longitude_x",
    "longitude_y",  # cópias de lat/lon
    "areia_total",  # areia + argila + silte = 1000 g/kg
    "temp_media",
    "temp_min_media",
    "umidade_media",
    "vento_medio_ms",  # colinearidade (legado)
}
