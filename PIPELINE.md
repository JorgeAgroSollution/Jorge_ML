# Pipeline de previsão agrícola

Este documento descreve o fluxo do projeto, as funções mais relevantes e o objetivo de cada etapa.

## Visão geral

O projeto estima e avalia a produtividade agrícola municipal usando:

- rendimento histórico da PAM/IBGE;
- localização dos municípios;
- características do solo;
- clima diário da NASA POWER;
- calendários fenológicos por cultura;
- validação espacial e temporal;
- modelos `HistGradientBoostingRegressor`.

Fluxo principal:

```text
PAM/IBGE
   + referência dos municípios
   + perfis de solo
   + clima diário NASA POWER
             ↓
Limpeza do alvo
             ↓
Features climáticas por cultura e janela fenológica
             ↓
Junção por município, cultura e ano
             ↓
Lag relativo de produtividade
             ↓
dataset_completo.csv
             ↓
Avaliação espacial e temporal
             ↓
Métricas, intervalos P10/P90 e importância das features
```

Os dois pontos de entrada são:

- [`build_dataset.py`](./build_dataset.py): coleta e monta o dataset;
- [`avaliar.py`](./avaliar.py): treina e avalia o modelo.

## 1. Configuração geral

O arquivo [`config.py`](./config.py) centraliza as premissas do pipeline:

- estado analisado: `SP`;
- período: 2000 a 2025;
- culturas analisadas;
- diretórios de dados brutos, processados, cache e resultados;
- calendários fenológicos;
- limites plausíveis de rendimento;
- regras de limpeza;
- colunas que não podem ser usadas como features.

As culturas configuradas são:

```text
soja, milho, feijão, arroz, trigo,
algodão, café, cana, laranja e mandioca
```

### Calendários fenológicos

A classe [`Janela`](./config.py) representa uma fase agrícola, como implantação,
floração, enchimento, maturação ou colheita.

A classe [`Calendario`](./config.py) reúne, para cada cultura:

- janelas fenológicas;
- coeficientes de cultura mensais (`Kc`);
- temperatura-base para graus-dia;
- limiar de estresse por calor;
- indicação de cultura perene.

Por exemplo, para soja, são consideradas aproximadamente:

- implantação: outubro-novembro do ano anterior;
- floração: dezembro-janeiro;
- enchimento: fevereiro-março.

### Regras de qualidade e prevenção de vazamento

O arquivo também define:

- limites de rendimento por cultura;
- primeiro ano válido para séries com quebra metodológica;
- área colhida mínima;
- quantidade máxima de anos consecutivos com rendimento idêntico;
- limites para detectar saltos anormais.

A lista `COLS_VAZAMENTO` exclui colunas que revelariam diretamente o alvo,
como `producao`, `rendimento`, `area_colhida`, `y_log`, `ano` e identificadores
do município.

## 2. Referência dos municípios

O módulo [`referencia.py`](./referencia.py) cria a tabela-base dos municípios.

### `normalizar_nome`

A função [`normalizar_nome`](./referencia.py) prepara nomes para junções
quando a PAM não fornece o código IBGE. Ela:

1. converte o texto para minúsculas;
2. remove acentos;
3. remove o sufixo `- SP`;
4. corrige alguns casos de codificação;
5. elimina espaços duplicados.

Exemplo:

```text
"São José do Rio Preto - SP"
→ "sao jose do rio preto"
```

O código IBGE continua sendo a chave principal. A normalização de nomes é
apenas um mecanismo de fallback.

### `construir_referencia`

A função [`construir_referencia`](./referencia.py) executa as seguintes ações:

1. verifica o cache `data/raw/municipios_ref.csv`;
2. se o cache existir, reutiliza-o;
3. caso contrário, consulta a malha municipal do IBGE;
4. obtém coordenadas distribuídas pelo `agrobr`;
5. filtra os municípios de São Paulo;
6. associa código IBGE, nome, latitude e longitude;
7. cria o nome normalizado;
8. salva a referência em cache.

## 3. Entrada e preparação da PAM

Essa etapa é coordenada por [`build_dataset.py`](./build_dataset.py).

### `carregar_pam`

A função [`carregar_pam`](./build_dataset.py) lê um CSV, remove a coluna
opcional `fonte` e verifica as colunas obrigatórias:

- `localidade`;
- `ano`;
- `produto`;
- `rendimento`;
- `area_colhida`;
- `area_plantada`;
- `producao`.

Uso:

```bash
python build_dataset.py --pam-csv data/raw/df_market_pam.csv
```

### `coletar_pam`

A função assíncrona [`coletar_pam`](./build_dataset.py) consulta a PAM
municipal por cultura através do `agrobr`.

Para cada cultura, ela:

1. consulta os anos de 2000 a 2025;
2. adiciona a coluna `produto`;
3. concatena os resultados;
4. renomeia `quantidade_produzida` para `producao`;
5. renomeia `rendimento_medio` para `rendimento`.

Uso:

```bash
python build_dataset.py --coletar-pam
```

### `ligar_codigo_ibge`

A função [`ligar_codigo_ibge`](./build_dataset.py) padroniza a chave municipal.

Se a PAM contém `localidade_cod`, ela é convertida para `cod_ibge`. Caso
contrário, o nome normalizado da localidade é associado à tabela de referência.

Linhas sem código IBGE são descartadas, pois não podem ser associadas com
segurança ao solo e ao clima.

## 4. Limpeza do alvo

O módulo [`alvo.py`](./alvo.py) trata a produtividade, que é a variável
explicada pelo modelo.

### `limpar_pam`

A função [`limpar_pam`](./alvo.py) aplica as regras de qualidade:

1. remove registros sem campos essenciais;
2. mantém apenas rendimento e área colhida positivos;
3. calcula indicadores de perda e consistência;
4. remove quebras conhecidas de série;
5. remove rendimentos fora dos limites agronômicos;
6. identifica saltos em relação à mediana municipal;
7. trata rendimentos repetidos;
8. cria o alvo logarítmico;
9. calcula o peso amostral.

São criadas, entre outras, as colunas:

- `razao_colhida_plantada`;
- `perda_flag`;
- `rendimento_calc`;
- `rend_repetido`;
- `y_log`;
- `peso`.

O alvo usado no modelo é:

```python
y_log = log(rendimento)
```

O peso é:

```python
peso = sqrt(area_colhida)
```

Rendimentos idênticos por pelo menos três anos não são removidos por padrão.
Eles recebem peso reduzido, pois podem representar estimativas repetidas.
Com `--descartar-repetidos`, esses registros são excluídos.

A função retorna:

1. o dataframe limpo;
2. um relatório auditável das linhas removidas por regra.

O relatório é salvo em:

```text
data/processed/relatorio_limpeza.csv
```

## 5. Tratamento do solo

O módulo [`solo.py`](./solo.py) transforma perfis da Embrapa em uma tabela
municipal.

### `tabela_solo_municipal`

A função [`tabela_solo_municipal`](./solo.py):

1. normaliza o município;
2. converte variáveis numéricas;
3. remove horizontes orgânicos;
4. usa horizontes superficiais `A` para química e fertilidade;
5. usa horizontes `A` e `B` até 100 cm para textura;
6. calcula profundidade máxima dos perfis;
7. agrega os dados por mediana;
8. registra a quantidade de perfis;
9. junta o resultado à referência municipal.

As propriedades químicas incluem, quando disponíveis:

- pH;
- carbono orgânico;
- CTC;
- saturação por bases;
- alumínio;
- fósforo.

As propriedades de textura incluem:

- argila;
- silte.

A areia não entra diretamente, pois é colinear com argila e silte.

### `imputar_solo_knn`

A função [`imputar_solo_knn`](./solo.py) preenche municípios sem perfil usando
os vizinhos geográficos mais próximos.

O processo:

1. identifica municípios com e sem perfil;
2. usa distância haversine via `BallTree`;
3. encontra até três vizinhos;
4. calcula a média dos vizinhos;
5. imputa apenas quando a distância média é de até 80 km.

Também cria:

- `solo_imputado`;
- `solo_dist_km`.

Com `--sem-knn`, apenas municípios com perfil próprio são mantidos.

### `adicionar_cad`

A função [`adicionar_cad`](./solo.py) calcula `cad_mm`, a capacidade de água
disponível do solo, chamando [`cad_mm`](./clima.py).

Esse valor será utilizado no balanço hídrico. Atualmente a CAD é estimada
por uma pedotransferência simplificada e deve ser validada com dados locais.

## 6. Coleta e processamento climático

O módulo [`clima.py`](./clima.py) transforma o clima diário em indicadores
agronômicos.

### `celula`

A função [`celula`](./clima.py) converte latitude e longitude para a célula
da grade NASA POWER/MERRA-2:

```text
aproximadamente 0,5° de latitude × 0,625° de longitude
```

Municípios na mesma célula compartilham a requisição climática.

### `coletar_diario`

A função assíncrona [`coletar_diario`](./clima.py):

1. verifica se o Parquet da célula está no cache;
2. lê o cache, quando disponível;
3. caso contrário, consulta a NASA POWER pelo `agrobr`;
4. salva a série em `data/cache`.

São coletados:

- temperatura média, máxima e mínima;
- precipitação;
- umidade relativa;
- radiação;
- vento;
- pressão;
- umidade do solo na zona radicular.

O período começa em 1999, um ano antes da PAM, porque algumas culturas
possuem janelas fenológicas iniciadas no ano anterior à colheita.

### `preparar_diario`

A função [`preparar_diario`](./clima.py):

1. converte as datas;
2. ordena a série;
3. remove datas duplicadas;
4. transforma sentinelas como `-999` em ausentes;
5. interpola lacunas de até cinco dias;
6. substitui precipitação ausente por zero;
7. impede precipitação negativa.

### Altitude

[`altitude_da_celula`](./clima.py) estima a altitude usando a pressão
atmosférica média da célula.

Essa altitude é aproximada e representa a célula climática, não necessariamente
a altitude exata do município.

## 7. Evapotranspiração e balanço hídrico

### `eto_fao56`

A função [`eto_fao56`](./clima.py) calcula a evapotranspiração de referência
usando Penman-Monteith FAO-56.

Ela combina:

- temperatura;
- umidade;
- radiação;
- vento;
- latitude;
- altitude;
- radiação extraterrestre.

O resultado é a demanda atmosférica de água em milímetros por dia.

### `balanco_hidrico`

A função [`balanco_hidrico`](./clima.py) simula um reservatório de água no solo.

Para cada dia, ela:

1. inicia o armazenamento em 50% da CAD;
2. adiciona a precipitação;
3. limita o armazenamento à CAD;
4. calcula o fator de estresse hídrico `Ks`;
5. calcula a evapotranspiração real `ETa`;
6. reduz o armazenamento.

Quando o solo está abastecido, `Ks = 1`. Quando a água fica escassa, `Ks`
diminui e a evapotranspiração real é reduzida.

## 8. Geração das features agrícolas

### `features_cultura`

A função [`features_cultura`](./clima.py) é uma das funções centrais do projeto.
Ela produz uma linha por ano, município e cultura.

O processo:

1. calcula a ETo diária;
2. aplica o `Kc` mensal da cultura;
3. calcula a evapotranspiração da cultura `ETc`;
4. calcula a evapotranspiração real `ETa`;
5. calcula estresse hídrico;
6. agrega o clima em cada janela fenológica.

Para cada janela são geradas features como:

- precipitação acumulada;
- temperatura máxima e mínima médias;
- radiação média;
- déficit hídrico;
- intensidade do estresse;
- dias de calor;
- dias de geada;
- maior sequência sem chuva;
- graus-dia;
- umidade média do solo.

Exemplos de nomes:

```text
floracao_prec
floracao_defic
floracao_estresse
floracao_dias_calor
floracao_seca_max
enchimento_gd
safra_prec
safra_defic
```

Se uma janela não tiver pelo menos 90% dos dias esperados, suas features
ficam como `NaN`, evitando fabricar uma cobertura climática inexistente.

### `obter_clima_features`

A função [`obter_clima_features`](./build_dataset.py) coordena a geração
climática:

1. agrupa a PAM por município;
2. identifica as culturas de cada município;
3. junta coordenadas e `cad_mm`;
4. determina a célula climática;
5. coleta o clima uma vez por célula;
6. calcula a altitude;
7. gera features para cada município, cultura e ano.

## 9. Montagem do dataset final

A função [`main`](./build_dataset.py) executa as etapas na seguinte ordem:

1. cria `data/processed`;
2. constrói a referência municipal;
3. carrega ou coleta a PAM;
4. liga os registros ao código IBGE;
5. aplica o filtro opcional de culturas;
6. limpa o alvo;
7. prepara a tabela de solo;
8. calcula as features climáticas;
9. junta PAM, clima e solo;
10. adiciona o lag de produtividade;
11. verifica a unicidade por município, cultura e ano;
12. grava os resultados.

As junções principais usam:

```text
cod_ibge + produto + ano
```

Arquivos gerados:

```text
data/processed/dataset_completo.csv
data/processed/dataset_<cultura>.csv
data/processed/relatorio_limpeza.csv
```

### `adicionar_lag`

A função [`adicionar_lag`](./build_dataset.py) cria `lag1_rel`.

Ela:

1. calcula a mediana anual de `y_log` por cultura;
2. calcula o desvio de cada município em relação à mediana do ano;
3. desloca esse desvio para o ano seguinte;
4. associa o valor ao mesmo município e cultura.

Assim, o lag representa a produtividade relativa do ano anterior, sem carregar
diretamente a tendência tecnológica geral.

O primeiro ano de cada município fica naturalmente sem esse valor.

## 10. Seleção das features

A função [`selecionar_features`](./avaliar.py):

1. seleciona apenas colunas numéricas;
2. remove colunas de vazamento;
3. remove colunas redundantes;
4. remove coordenadas por padrão;
5. pode remover `lag1_rel`;
6. remove colunas com mais de 50% de valores ausentes;
7. remove colunas constantes.

As features são principalmente:

- indicadores climáticos por janela;
- propriedades de solo;
- CAD;
- altitude;
- eventualmente `lag1_rel`.

## 11. Modelo de machine learning

### `_hgb`

A função [`_hgb`](./avaliar.py) cria um
`HistGradientBoostingRegressor` com:

- profundidade máxima 3;
- taxa de aprendizado 0,05;
- até 250 iterações;
- pelo menos 25 observações por folha;
- regularização L2;
- semente aleatória fixa.

O modelo é usado para:

1. previsão central;
2. quantil de 10%;
3. quantil de 90%.

## 12. Tendência tecnológica e previsão

### `Tendencia`

A classe [`Tendencia`](./alvo.py) representa a tendência tecnológica da
produtividade.

O método `fit`:

1. calcula a mediana anual de `y_log`;
2. ajusta uma reta sobre as medianas;
3. representa mudanças gerais de produtividade ao longo do tempo.

O método `predict` calcula a tendência esperada para novos anos.

Essa tendência é ajustada somente no treinamento de cada fold, evitando
vazamento temporal.

### `_ajustar_prever`

A função [`_ajustar_prever`](./avaliar.py):

1. ajusta a tendência no treino;
2. calcula a anomalia:

   ```text
   anomalia = y_log - tendência(ano)
   ```

3. cria o baseline baseado apenas na tendência;
4. cria o baseline município + tendência;
5. treina o modelo para prever a anomalia;
6. converte a previsão de volta para kg/ha;
7. calcula os intervalos P10/P90.

A previsão central é:

```text
modelo = exp(tendência + anomalia_predita)
```

## 13. Intervalos P10/P90

O pipeline treina modelos de quantil para obter:

- limite inferior de 10%;
- limite superior de 90%.

Uma parte dos municípios de treinamento é usada para calibração conformal.
O erro observado nessa calibração ajusta os limites finais.

Os resultados são:

```text
p10 = cenário inferior
p90 = cenário superior
```

O intervalo P10-P90 deve conter aproximadamente 80% das observações.

## 14. Validação espacial

A função [`avaliar_cultura`](./avaliar.py) avalia cada cultura separadamente.

Na validação espacial:

1. municípios são agrupados por latitude e longitude usando `KMeans`;
2. os grupos espaciais são usados no `GroupKFold`;
3. regiões inteiras ficam fora do treinamento;
4. o modelo é testado em regiões novas.

Isso evita uma avaliação excessivamente otimista causada por municípios vizinhos
no treino e no teste, especialmente porque municípios próximos compartilham
clima e células NASA.

O resultado é identificado como:

```text
espacial (regiões novas)
```

## 15. Validação temporal

Também é feita validação temporal com janela expansiva:

```text
treinar com o passado → testar nos três anos seguintes
```

Esse procedimento simula melhor a utilização real do modelo para prever anos
futuros.

O resultado é identificado como:

```text
temporal (anos futuros)
```

## 16. Métricas

A função [`_metricas`](./avaliar.py) calcula:

- `MAE_tendencia`;
- `MAE_mun+tend`;
- `MAE_modelo`;
- `skill_vs_tendencia`;
- `skill_vs_mun+tend`;
- `R2_anomalia`;
- proporção abaixo do P10;
- proporção dentro do intervalo P10-P90.

### Skill

```text
skill_vs_tendencia =
1 - MAE_modelo / MAE_tendencia
```

Interpretação:

- valor positivo: o modelo melhora sobre a tendência;
- valor zero: o modelo não agrega valor;
- valor negativo: o modelo é pior que a tendência.

### Cobertura dos intervalos

`dentro_P10_P90` mede a proporção de observações dentro do intervalo.
O valor esperado é aproximadamente 0,80.

## 17. Importância das features

O pipeline usa importância por permutação:

1. mede o erro normal;
2. embaralha uma feature;
3. mede o aumento do erro;
4. usa esse aumento como importância.

Os resultados são salvos em:

```text
resultados/importancia_permutacao.csv
```

As métricas são salvas em:

```text
resultados/metricas.csv
```

## 18. Comandos de uso

### Construir o dataset usando uma PAM existente

```bash
python build_dataset.py --pam-csv data/raw/df_market_pam.csv
```

### Coletar a PAM pelo `agrobr`

```bash
python build_dataset.py --coletar-pam
```

### Não imputar solo por KNN

```bash
python build_dataset.py --pam-csv data/raw/df_market_pam.csv --sem-knn
```

### Processar somente algumas culturas

```bash
python build_dataset.py \
  --pam-csv data/raw/df_market_pam.csv \
  --culturas soja milho
```

### Avaliar o dataset completo

```bash
python avaliar.py --dados data/processed/dataset_completo.csv
```

### Avaliar culturas específicas

```bash
python avaliar.py \
  --dados data/processed/dataset_completo.csv \
  --culturas soja milho
```

### Avaliar um CSV legado

```bash
python avaliar.py --dados df_merged.csv --legado
```

### Remover o lag

```bash
python avaliar.py \
  --dados data/processed/dataset_completo.csv \
  --sem-lag
```

### Descartar rendimentos repetidos

```bash
python avaliar.py \
  --dados data/processed/dataset_completo.csv \
  --descartar-repetidos
```

## 19. Testes

Os testes estão em [`test_pipeline.py`](./test_pipeline.py) e verificam:

- plausibilidade da ETo;
- redução da evapotranspiração durante a seca;
- tratamento de janelas sem cobertura;
- remoção de quebras e outliers;
- ajuste da tendência;
- ausência de vazamento no `lag1_rel`;
- recuperação de sinal climático em dados sintéticos.

Executar:

```bash
python -m pytest -q test_pipeline.py
```

No ambiente em que esta documentação foi elaborada, a coleta dos testes não
foi concluída porque o interpretador não tinha `scikit-learn` instalado:

```text
ModuleNotFoundError: No module named 'sklearn'
```

A dependência está declarada em [`pyproject.toml`](./pyproject.toml) e
[`requirements.txt`](./requirements.txt).

## 20. Limitações conhecidas

### Calendários agrícolas

Os calendários de `config.py` são aproximações e devem ser revisados com
especialistas, ZARC, CATI ou IAC.

Isso é especialmente importante para:

- milho safrinha;
- feijão com múltiplas safras;
- café;
- cana;
- laranja;
- mandioca.

### Capacidade de água disponível

A `cad_mm` usa uma pedotransferência simplificada. O ideal é validar ou
substituir esse cálculo por dados medidos ou mapas de retenção de água.

### Resolução climática

A NASA POWER trabalha com uma grade relativamente ampla. Por isso, a validação
espacial agrupa regiões próximas para evitar que o modelo seja avaliado em
municípios climaticamente quase idênticos aos usados no treinamento.

### Formato de dados externos

As consultas do `agrobr` podem exigir ajustes caso a versão instalada altere
nomes ou formatos das colunas retornadas.

### Modelo não persistido

O script `avaliar.py` treina os modelos durante a execução, mas não salva um
arquivo de modelo para inferência posterior. Ele salva apenas métricas e
importâncias.

## Resumo

O pipeline:

1. obtém a PAM;
2. padroniza municípios por código IBGE;
3. agrega perfis de solo;
4. calcula a capacidade de água disponível;
5. coleta clima diário da NASA POWER;
6. calcula ETo pela FAO-56;
7. simula o balanço hídrico;
8. agrega o clima por fase fenológica;
9. limpa o rendimento;
10. cria o alvo logarítmico e o lag relativo;
11. junta PAM, solo e clima;
12. treina um modelo para prever anomalias de produtividade;
13. avalia regiões novas e anos futuros;
14. calcula intervalos P10/P90;
15. gera métricas e importância das features.
