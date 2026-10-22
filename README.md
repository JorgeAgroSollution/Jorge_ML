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
| `test_pipeline.py` | Testes, inclusive recuperação de sinal em dados sintéticos. |

## Uso
```bash
pip install -r requirements.txt
python build_dataset.py --pam-csv data/raw/df_market_pam.csv     # ou --coletar-pam
python avaliar.py --dados data/processed/dataset_completo.csv
python avaliar.py --dados df_merged.csv --legado                  # avalia o CSV antigo com o método novo
python -m pytest -q test_pipeline.py
```

## O que foi testado aqui (sem rede)
- Física (ETo, balanço hídrico), janelas, limpeza, tendência, lag sem vazamento e avaliação: `test_pipeline.py` passa.
- Avaliação aplicada ao `df_merged.csv` antigo: ver `resultados/metricas.csv`.

## O que NÃO foi testado (requer rede/dados reais) — validar na primeira execução
1. `coletar_diario` (agrobr NASA POWER diário com `parameters=[... 'PS','GWETROOT']`) e `construir_referencia` (malha IBGE + geopandas).
2. `coletar_pam`: o formato devolvido pelo agrobr pode exigir ajuste de nomes de colunas (use `--pam-csv` para reaproveitar o arquivo atual).
3. `solo.tabela_solo_municipal`: assume `horizonte` com prefixo A/B/O e `profundidade` em cm. Conferir.
4. `cad_mm` usa pedotransferência simples; trocar por CAD medida/mapeada (`agua_dispo` — validar unidade).
5. Calendários e Kc em `config.py` são aproximações.
