# Copilot instructions for `agro_ml`

## Project scope

This repository builds and evaluates agricultural-yield models for municipalities in
São Paulo, Brazil. The pipeline combines IBGE PAM yield data, municipality reference
data, Embrapa soil profiles, and daily NASA POWER weather data.

## Environment and commands

- Use Python `>=3.13` and use `uv` as the preferred dependency and environment
  manager. The dependency set is declared in `pyproject.toml` and locked in
  `uv.lock`; `requirements.txt` is provided only as a pip-based fallback.
- Create or synchronize the project environment with `uv sync`. Run project commands
  through the managed environment with `uv run`, for example `uv run python
  build_dataset.py ...` and `uv run pytest ...`. Update dependencies with `uv add
  <package>` or `uv remove <package>` so that `pyproject.toml` and `uv.lock` stay
  consistent. Use `pip install -r requirements.txt` only when `uv` is unavailable.
- Build the processed dataset from an existing PAM export:

  ```bash
  uv run python build_dataset.py --pam-csv data/raw/df_market_pam.csv
  ```

- Fetch PAM through `agrobr` instead:

  ```bash
  uv run python build_dataset.py --coletar-pam
  ```

- Restrict dataset construction to selected crops with `--culturas`; use
  `--sem-knn` to exclude municipalities without an observed soil profile rather than
  imputing from nearby municipalities.
- Evaluate a generated dataset:

  ```bash
  uv run python avaliar.py --dados data/processed/dataset_completo.csv
  ```

  `avaliar.py` also supports `--legado` for the older CSV format, `--culturas`,
  `--com-coordenadas`, `--sem-lag`, `--descartar-repetidos`, and `--sem-limpeza`.
- Run the pipeline tests:

  ```bash
  uv run pytest -q test_pipeline.py
  ```

- Run one test by node id, for example:

  ```bash
  uv run pytest -q test_pipeline.py::test_lag_relativo_nao_vaza_futuro
  ```

There is no configured lint command or lint configuration in this repository. The
tests are designed to run without network access; dataset collection and real-data
validation require the external services and source files described below.

## Architecture and data flow

1. `config.py` is the central domain configuration. It defines the São Paulo scope,
   crop list, year range, agronomic plausibility limits, series breaks, and
   phenological calendars/Kc values. Treat these values as explicit agronomic
   assumptions, not generic constants.
2. `referencia.py` creates or reads the cached municipality reference
   (`data/raw/municipios_ref.csv`), including IBGE codes, normalized names, and
   representative coordinates. Names are only a fallback join key; downstream joins
   should use `cod_ibge`.
3. `alvo.py` cleans PAM observations and creates the modeling target. It removes
   invalid/out-of-scope yields, produces an auditable cleaning report, logs yield in
   `y_log`, and assigns area-based/repeated-value weights. `Tendencia` models
   technology trend and must be fitted inside each training fold.
4. `solo.py` aggregates Embrapa soil horizons by municipality: chemistry uses
   surface horizons, texture uses the 0–100 cm range, and missing municipalities can
   receive marked nearest-neighbor imputation. `adicionar_cad` derives the available
   water capacity used by the water balance.
5. `clima.py` normalizes daily weather, computes FAO-56 reference
   evapotranspiration and a reservoir water balance, then emits one row per
   crop/year with features for the configured phenological windows. NASA POWER data
   is requested once per shared grid cell and cached as Parquet.
6. `build_dataset.py` orchestrates the asynchronous reference/PAM/soil/weather
   collection, joins by IBGE code and crop/year, adds the relative previous-year
   yield feature, and writes `data/processed/dataset_completo.csv`,
   crop-specific datasets, and `relatorio_limpeza.csv`.
7. `avaliar.py` selects numeric, non-leaking features and evaluates each crop with
   spatial grouped folds and expanding temporal windows. It compares against trend
   and municipality-plus-trend baselines, reports yield/error and anomaly metrics,
   and writes `resultados/metricas.csv` plus permutation importance. Prediction
   intervals use quantile models with conformal calibration.

## Repository-specific conventions

- Preserve the temporal boundary: features and the relative lag must not use future
  observations, and trend fitting must happen separately inside every evaluation
  training fold.
- Keep target leakage exclusions centralized in `config.py` (`COLS_VAZAMENTO` and
  `COLS_REDUNDANTES`) and use `selecionar_features` rather than manually adding
  production, area, yield, year, or identifier columns to models.
- Keep data collection imports lazy where they are currently lazy. Importing the
  transformation and test modules should not require a network request.
- Preserve auditability: cleaning rules belong in `alvo.limpar_pam`, and changes to
  filtering or imputation should retain/report the relevant flags and counts.
- Keep joins keyed by `cod_ibge`; use `normalizar_nome` only to bridge source files
  that do not contain an IBGE code.
- Treat incomplete weather windows as missing features. Do not silently fill a
  missing phenological window with a partial season; `features_cultura` requires
  adequate date coverage.
- Weather and soil assumptions are intentionally visible: NASA sentinel values are
  normalized in `preparar_diario`, soil KNN imputations are marked with
  `solo_imputado`/`solo_dist_km`, and the simple CAD estimate is documented as an
  approximation.
- Keep the model evaluation honest: spatial groups account for the coarse NASA grid,
  temporal splits represent future years, and a non-positive skill versus the trend
  baseline means the model adds no demonstrated value.
- Update `PIPELINE.md` or `README.md` when changing command-line behavior, data
  sources, output files, or agronomic assumptions. Review `config.py` and the
  corresponding tests when changing calendars, thresholds, feature names, or target
  cleaning.
- Generated caches and Parquet files under `data/` are ignored by Git. Avoid
  committing downloaded datasets, virtual environments, or generated Python files.
