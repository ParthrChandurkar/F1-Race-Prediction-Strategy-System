# F1 Race Prediction and Strategy System

An interactive Streamlit application for exploring historical Formula 1 data, predicting race outcomes, running Monte Carlo race simulations, and comparing weather-aware tyre and pit-stop strategies.

## Features

- Finishing-position and Top 10 prediction using saved scikit-learn artifacts
- Win, podium, Top 10, DNF, and average-finish estimates from repeated simulations
- Circuit- and weather-aware tyre strategy comparison
- Live race-engineer status based on the current lap and completed stops
- Model metrics, feature importance, driver, team, and historical analysis views
- Reproducible data and training stages defined in DVC

## Architecture

`app.py` provides the Streamlit interface. Runtime predictions are implemented in `src/predictor.py`, simulations in `src/simulator.py`, and strategy logic in `src/strategy.py`. Saved encoders, scalers, estimators, metrics, and metadata are stored under `models/`.

The training path loads raw Formula 1 CSV files, preprocesses and engineers features, trains several classification and regression candidates, evaluates them, and records model metadata. `dvc.yaml`, `dvc.lock`, and `params.yaml` define the reproducible pipeline. See [`docs/architecture.md`](docs/architecture.md) and [`docs/strategy-engine.md`](docs/strategy-engine.md).

## Tech Stack

- Python, pandas, NumPy, and scikit-learn
- Streamlit and Plotly
- DVC and YAML-based pipeline parameters
- pytest
- Docker and Docker Compose
- GitHub Actions

## Getting Started

Prerequisites: Python 3.11 or newer and pip.

```bash
python -m venv .venv
```

Activate the environment, install dependencies, and start the application:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS or Linux, activate with `source .venv/bin/activate`. The application is normally available at `http://localhost:8501`. Checked-in model artifacts support inference without retraining.

## Data and Training

The training pipeline expects the Formula 1 CSV tables listed in `data/raw/PLACE_CSV_FILES_HERE.txt`. The README previously referenced the [Formula 1 World Championship dataset on Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020); verify the dataset license and current file names before downloading.

Reproduce the pipeline with:

```bash
dvc repro
```

Pipeline outputs are written to `data/processed/`, `models/`, and the local model registry under `mlops/model_registry/`.

## Docker

Run the Streamlit application:

```bash
docker compose up --build f1-app
```

Run the optional training service:

```bash
docker compose --profile train up --build f1-train
```

The Compose file mounts the data, model, experiment, and registry paths so generated artifacts remain available on the host.

## Testing

```bash
python -m pytest -q
```

The tests cover data loading, required model artifacts, and prediction behavior. No test-result count is claimed here because results depend on the checked-out environment and artifacts.

## Limitations

- Predictions and strategies are analytical estimates, not guarantees of race outcomes.
- Future-grid constants and circuit assumptions are stored in source and require manual updates.
- Historical data may not capture regulation, driver, team, weather, or car-performance changes.
- Although Docker, DVC, and CI files are present, this repository does not contain Kubernetes manifests, EC2 provisioning, or an S3-backed DVC remote configuration.

