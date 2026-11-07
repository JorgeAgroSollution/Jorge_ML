# agro_ml — pipeline refatorado (produtividade agrícola, pequeno produtor paulista)

## Estrutura
| Arquivo | Função |
|---|---|
| `config.py` | Calendários fenológicos por cultura, limites de plausibilidade, quebras de série da PAM. **Premissas a revisar com agrônomo/CATI.** |
| `referencia.py` | Tabela de municípios (código IBGE, centroide) e normalização de nomes. |
| `solo.py` | Solo por município sem misturar horizontes (química do horizonte A, textura 0–100 cm), mantém P, KNN opcional para municípios sem perfil (marcado), CAD. |
| `clima.py` | NASA POWER **diário** → ETo (FAO-56), balanço hídrico, features por janela fenológica (déficit, veranico, geada, calor, graus-dia…). |
| `alvo.py` | Limpeza auditável do rendimento (PAM) + tendência tecnológica (ajustada só no treino). |
| `build_dataset.py` | Coleta e montagem ponta a ponta, junção por código IBGE, uma tabela por cultura. |
| `avaliar.py` | Validação espacial e temporal, baseline, quantis P10/P90 calibrados (conformal), importância por permutação. |
| `modelar_culturas.py` | Compara ElasticNet, RandomForest e HistGradientBoosting por cultura (alvo = anomalia sobre a tendência, com skill vs. regras simples), treina o vencedor e grava no artefato tendência, faixa P10–P90, skill e `perfil_local`. |
| `predicao.py` | Previsão com o mínimo de dados do agricultor (cultura, ano, latitude, longitude; solo opcional), usando só o que está no artefato. |
| `test_pipeline.py` | Testes, inclusive recuperação de sinal em dados sintéticos. |
| `test_predicao.py` | Testes do treino com artefatos fixos e da previsão por latitude/longitude. |

## Uso
As dependências ficam em grupos no `pyproject.toml`:

| Grupo | Para quê | Instalação |
|---|---|---|
| principal | só a API (fastapi, uvicorn, pandas, numpy, pyarrow, scikit-learn 1.9.1, joblib) | `uv sync --no-dev` ou `pip install -r requirements.txt` |
| `dev` | testes (pytest, httpx) | `uv sync` (padrão) |
| `pipeline` | coleta e treino (agrobr, geopandas) | `uv sync --all-groups` |
| `notebook` | análise exploratória em `EDA/` (matplotlib, seaborn, ipykernel, jupytext) | `uv sync --all-groups` |

O scikit-learn fica fixo na versão que gerou os `.joblib`; trocar a versão exige treinar de novo.

```bash
uv sync --all-groups
python build_dataset.py --pam-csv data/raw/df_market_pam.csv     # ou --coletar-pam
python avaliar.py --dados data/processed/dataset_completo.csv
python avaliar.py --dados df_merged.csv --legado                  # avalia o CSV antigo com o método novo
python -m pytest -q test_pipeline.py
uv run python modelar_culturas.py                                # todas as culturas (gera os artefatos usados pela API)
uv run python modelar_culturas.py --culturas milho cafe
```

## API de predição

A API FastAPI carrega os artefatos em `resultados/modelos_cultura/` e o artefato
legado `resultados/modelo_soja_final.joblib`. Ela recebe dados brutos locais (clima
diário, solo e município), calcula as features fenológicas do pipeline e seleciona
automaticamente o modelo pela cultura:

```bash
uv sync
uv run uvicorn api:app --reload
```

A documentação interativa fica em `http://127.0.0.1:8000/docs`; o catálogo está em
`GET /modelos` e a predição em `POST /predicoes`. O corpo da predição contém
`cultura`, `ano`, `municipio`, `solo`, `lag1_rel` e `clima_diario`. Cada item de
`clima_diario` exige `data`, temperaturas, precipitação, umidade, radiação, vento,
pressão e `umidade_solo_raiz`. A resposta informa o modelo usado e o
`rendimento_predito_kg_ha`.

### Contrato dos endpoints

Todos os endpoints retornam JSON. Campos extras são rejeitados e erros de entrada
retornam `422 Unprocessable Entity` com detalhes de validação. O endpoint
`/health` retorna `200` quando a aplicação está respondendo:

```json
{
  "status": "ok",
  "modelos_disponiveis": 9
}
```

`GET /modelos` lista as culturas que podem ser previstas. O campo `features`
informa as variáveis que serão usadas internamente pelo modelo:

```json
[
  {
    "cultura": "milho",
    "modelo": "HistGradientBoostingRegressor",
    "features": ["chuva_floracao_mm", "cad_mm", "lag1_rel"],
    "artefato": "resultados/modelos_cultura/modelo_milho.joblib",
    "unidade_predicao": "kg/ha"
  }
]
```

`POST /predicoes/simples` é recomendado quando o consumidor possui apenas a
cultura, o ano e as coordenadas. O serviço usa o clima típico e o perfil local
armazenados no artefato; `solo` e `lag1_rel` podem ser informados para substituir
essas referências:

```json
{
  "cultura": "milho",
  "ano": 2026,
  "latitude": -21.17,
  "longitude": -47.81,
  "solo": {
    "ph_h2o": 5.5,
    "argila": 350,
    "prof_max_cm": 100
  },
  "lag1_rel": 0.02
}
```

Exemplo de resposta:

```json
{
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
  "confiavel": true,
  "avisos": [],
  "modelo": "HistGradientBoostingRegressor"
}
```

`POST /predicoes` é a opção para consumidores que possuem dados observados.
Além de `cultura` e `ano`, envie `municipio`, `solo`, `lag1_rel` e a série
`clima_diario`. Cada registro climático usa data `AAAA-MM-DD`, temperatura em
°C, precipitação em mm, radiação em MJ/m²/dia, vento em m/s e pressão em kPa.
A lista deve cobrir todos os dias necessários às janelas fenológicas da cultura,
sem enviar apenas uma amostra parcial:

```json
{
  "cultura": "milho",
  "ano": 2026,
  "municipio": {
    "cod_ibge": 3500105,
    "latitude": -22.3,
    "longitude": -48.3
  },
  "solo": {
    "ph_h2o": 5.5,
    "carbono_organico": 15,
    "ctc": 8,
    "saturacao_bases": 50,
    "aluminio": 2,
    "argila": 350,
    "silte": 250,
    "prof_max_cm": 100
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
      "umidade_solo_raiz": 0.6
    }
  ]
}
```

A resposta da predição completa identifica o município, o modelo e as features
usadas:

```json
{
  "cultura": "milho",
  "cod_ibge": 3500105,
  "ano": 2026,
  "rendimento_predito_kg_ha": 5840.2,
  "modelo": "HistGradientBoostingRegressor",
  "features_utilizadas": ["chuva_floracao_mm", "cad_mm", "lag1_rel"]
}
```

Para consultar o contrato gerado automaticamente, abra `/docs` (Swagger UI),
`/redoc` (ReDoc) ou `/openapi.json` após iniciar a API.

Os caminhos podem ser sobrescritos com `AGRO_ML_MODELOS_DIR` e
`AGRO_ML_MODELO_SOJA`. A API não faz coleta externa durante a requisição; os dados
diários devem ser preparados pelo consumidor ou por uma etapa de coleta separada.

`modelar_culturas.py` gera um relatório de qualidade em
`resultados/relatorio_analise_culturas.csv`, as métricas por modelo em
`resultados/metricas_modelos.csv`, um artefato por cultura em
`resultados/modelos_cultura/` e as previsões ajustadas em
`resultados/previsoes_culturas.csv`. A escolha do modelo usa o menor MAE em
kg/ha médio na validação temporal; as métricas espacial e logarítmica também
ficam registradas para comparação.

### Previsão com o mínimo de dados do agricultor

`POST /predicoes/simples` pede só `cultura`, `ano`, `latitude` e `longitude`. O solo do talhão é opcional
(qualquer campo informado substitui o do município). Não é preciso enviar clima diário, código IBGE nem
`lag1_rel`:

```bash
curl -X POST http://127.0.0.1:8000/predicoes/simples -H 'Content-Type: application/json' \
  -d '{"cultura": "milho", "ano": 2026, "latitude": -21.17, "longitude": -47.81}'
```

A resposta traz o rendimento previsto, a faixa `p10_kg_ha`–`p90_kg_ha` ("ano ruim" a "ano bom"), a
`referencia_municipio_kg_ha` (tendência + média histórica do município), de onde veio o solo e o histórico, um
indicador `confiavel` e `avisos`.

O que é calculado **uma vez, no treino**, e reaproveitado (dentro de `modelo_<cultura>.joblib`):

| Chave do artefato | Conteúdo |
|---|---|
| `tendencia_coef`, `ano_ultimo` | Reta de ganho tecnológico (log-rendimento) ajustada com todos os dados. |
| `intervalo_log` | Quantis 10% e 90% do erro fora da amostra (o mais largo entre validação espacial e temporal). |
| `validacao` | MAE e skill (espacial/temporal) vs. tendência e vs. tendência + média do município, inclusive **sem o rendimento do ano anterior** (horizonte ≥ 2 anos). `predicao.confiavel` exige ganho ≥ `SKILL_MINIMO` sobre a melhor regra simples da situação. |
| `perfil_local` | Por município: clima típico por janela, altitude, solo, CAD, último `lag1_rel` e média histórica. |

Limites assumidos (ver `config.py` e `predicao.py`): o clima da safra futura não é conhecido, então usa-se o
clima típico; o município usado é o mais próximo **com dado da cultura** (aviso acima de 50 km, recusa acima de
150 km); anos além de `ano_ultimo + 3` só extrapolam a tendência. Artefatos gerados antes desta versão não têm
`perfil_local`: rode `uv run python modelar_culturas.py` de novo.

## O que foi testado aqui (sem rede)
- Física (ETo, balanço hídrico), janelas, limpeza, tendência, lag sem vazamento e avaliação: `test_pipeline.py` passa.
- Avaliação aplicada ao `df_merged.csv` antigo: ver `resultados/metricas.csv`.

## O que NÃO foi testado (requer rede/dados reais) — validar na primeira execução
1. `coletar_diario` (agrobr NASA POWER diário com `parameters=[... 'PS','GWETROOT']`) e `construir_referencia` (malha IBGE + geopandas).
2. `coletar_pam`: o formato devolvido pelo agrobr pode exigir ajuste de nomes de colunas (use `--pam-csv` para reaproveitar o arquivo atual).
3. `solo.tabela_solo_municipal`: assume `horizonte` com prefixo A/B/O e `profundidade` em cm. Conferir.
4. `cad_mm` usa pedotransferência simples; trocar por CAD medida/mapeada (`agua_dispo` — validar unidade).
5. Calendários e Kc em `config.py` são aproximações.
