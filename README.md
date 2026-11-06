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
| `modelar_culturas.py` | Compara ElasticNet, RandomForest e HistGradientBoosting por cultura, treina o vencedor e salva previsões. |
| `test_pipeline.py` | Testes, inclusive recuperação de sinal em dados sintéticos. |

## Uso
```bash
pip install -r requirements.txt
python build_dataset.py --pam-csv data/raw/df_market_pam.csv     # ou --coletar-pam
python avaliar.py --dados data/processed/dataset_completo.csv
python avaliar.py --dados df_merged.csv --legado                  # avalia o CSV antigo com o método novo
python -m pytest -q test_pipeline.py
uv run python modelar_culturas.py                                # todas as culturas, exceto soja
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

## O que foi testado aqui (sem rede)
- Física (ETo, balanço hídrico), janelas, limpeza, tendência, lag sem vazamento e avaliação: `test_pipeline.py` passa.
- Avaliação aplicada ao `df_merged.csv` antigo: ver `resultados/metricas.csv`.

## O que NÃO foi testado (requer rede/dados reais) — validar na primeira execução
1. `coletar_diario` (agrobr NASA POWER diário com `parameters=[... 'PS','GWETROOT']`) e `construir_referencia` (malha IBGE + geopandas).
2. `coletar_pam`: o formato devolvido pelo agrobr pode exigir ajuste de nomes de colunas (use `--pam-csv` para reaproveitar o arquivo atual).
3. `solo.tabela_solo_municipal`: assume `horizonte` com prefixo A/B/O e `profundidade` em cm. Conferir.
4. `cad_mm` usa pedotransferência simples; trocar por CAD medida/mapeada (`agua_dispo` — validar unidade).
5. Calendários e Kc em `config.py` são aproximações.
