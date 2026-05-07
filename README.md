# 🏎️ F1 Race Prediction & Strategy System — 2025 Season
### ML-Powered · Future Race Predictions · Full Strategy Engine · Monte Carlo Simulation

---

## Project Overview

A professional Formula 1 prediction system that uses machine learning trained on
historical race data (2000–2024) to predict future 2025 season race outcomes.

**What it does:**
- Predicts finishing positions for all 20 current F1 drivers at any 2025 circuit
- Predicts qualifying order and pole position probability
- Runs Monte Carlo race simulations (up to 2000 iterations)
- Provides full pit stop strategy with tyre compounds, pit windows, and safety car analysis
- Driver and team historical analysis (2014–2024)
- Full ML model comparison dashboard

**2025 Grid Included:**
All 20 current drivers including Hamilton→Ferrari, Antonelli→Mercedes,
Sainz→Williams, Lawson→Racing Bulls, Doohan→Alpine

---

## Dataset

Source: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

**Files needed in data/raw/:**
```
results.csv  |  races.csv  |  drivers.csv  |  constructors.csv
qualifying.csv  |  pit_stops.csv  |  lap_times.csv  |  circuits.csv
```

---

## Folder Structure

```
f1-ml-project/
├── app.py                        ← Streamlit web app (8 pages)
├── requirements.txt
├── params.yaml                   ← DVC parameters
├── dvc.yaml                      ← DVC pipeline
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── install.ps1                   ← Windows SSL-safe installer
├── install.bat
├── SETUP_WINDOWS.md
│
├── src/
│   ├── f1_2024_data.py           ← 2025 driver grid, circuits, ratings
│   ├── data_loader.py            ← Load and merge 8 CSVs
│   ├── preprocessing.py          ← Clean, encode, scale
│   ├── feature_engineering.py   ← Rolling averages, targets
│   ├── train_models.py           ← Train all 9 ML models
│   ├── evaluate_models.py        ← Print evaluation report
│   ├── predictor.py              ← Future race predictions
│   ├── simulator.py              ← Monte Carlo simulation
│   └── strategy.py               ← Full strategy engine
│
├── ml_pipeline/                  ← DVC stage scripts
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
│
├── mlops/
│   ├── model_registry/
│   │   ├── register_model.py
│   │   └── registry.json
│   └── experiments/
│
├── tests/
│   ├── test_data_loading.py
│   ├── test_model_files.py
│   └── test_prediction.py
│
├── docs/
│   └── architecture.md
│
├── data/
│   ├── raw/                      ← Place Kaggle CSVs here
│   └── processed/                ← Auto-generated
│
└── models/                       ← Auto-generated after training
```

---

## Installation

### Step 1 — Create virtual environment

```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### Step 2 — Install dependencies

**Normal networks:**
```bash
pip install -r requirements.txt
```

**College / corporate network (SSL error fix):**
```powershell
# Windows
.\install.ps1

# OR manual:
pip install scikit-learn numpy pandas joblib streamlit plotly dvc pyyaml pytest matplotlib seaborn --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org
```

**Permanent SSL fix — create C:\Users\YourName\pip\pip.ini:**
```ini
[global]
trusted-host =
    pypi.org
    files.pythonhosted.org
    pypi.python.org
```

---

## Running the Project

### Step 1 — Place CSV files
Copy the 8 Kaggle CSV files into `data/raw/`

### Step 2 — Train models
```bash
python src/train_models.py
```

Expected output:
```
============================================================
  F1 ML PROJECT — MODEL TRAINING
============================================================
[1/6] Loading raw CSVs ...
      Master shape: (26000, 35)
[2/6] Cleaning ...
[3/6] Encoding categoricals ...
[4/6] Engineering features ...
[5/6] Splitting train/test ...
      Train: (19600, 11)  Test: (4900, 11)
[6/6] Training models ...
  >> Logistic Regression   Accuracy=0.84  F1=0.81
  >> Decision Tree         Accuracy=0.82  F1=0.80
  >> Random Forest         Accuracy=0.87  F1=0.85
  >> SVM                   Accuracy=0.85  F1=0.83
  >> Naive Bayes            Accuracy=0.76  F1=0.73
  >> Linear Regression     MAE=2.8  R2=0.71
  >> Ridge Regression      MAE=2.7  R2=0.72
  >> Lasso Regression      MAE=2.9  R2=0.70
  >> K-Means Clustering    Inertia=...
  ALL MODELS TRAINED AND SAVED
```

### Step 3 — Launch the app
```bash
streamlit run app.py
```

Open: **http://localhost:8501**

---

## UI Pages

| Page | What it does |
|---|---|
| 🏠 Dashboard | 2025 driver grid, team ratings, system overview |
| 🏁 Race Prediction | Predict full 20-driver race result for any 2025 circuit |
| 🥇 Qualifying Prediction | Predict qualifying order and pole position |
| 🎲 Race Simulation | Monte Carlo simulation with win/podium/DNF probabilities |
| 🛞 Strategy Centre | Full pit stop strategy, tyre compounds, pit windows, SC analysis |
| 👤 Driver Analysis | 10-year historical stats, circuit form, skill comparison |
| 🏭 Team Analysis | Constructor trends, driver stats, 2025 lineup |
| 📊 Model Performance | All 9 model metrics, confusion matrix, feature importance |

---

## Running DVC Pipeline

```bash
# Initialize (first time only)
dvc init -f

# Run full pipeline
dvc repro

# Check status
dvc status

# Change a parameter and re-run (only affected stages re-run)
# Edit params.yaml -> change n_estimators: 200
dvc repro
```

---

## Running Docker

```bash
# Train models locally first, then:
docker build -t f1-ml-app .
docker-compose up --build

# Open http://localhost:8501

# Stop
docker-compose down
```

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## GitHub CI/CD

Push to GitHub and CI automatically:
1. Checks all 17 source files exist
2. Compiles all Python modules
3. Validates params.yaml and dvc.yaml
4. Runs simulator and strategy unit tests
5. Validates model registry

```bash
git init
git add .
git commit -m "F1 ML Project - Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/f1-ml-project.git
git branch -M main
git push -u origin main
```

---

## ML Models

### Classification (predicts: will driver finish Top 10?)
| Model | Accuracy | F1 |
|---|---|---|
| Random Forest | ~87% | ~0.85 |
| SVM | ~85% | ~0.83 |
| Logistic Regression | ~84% | ~0.81 |
| Decision Tree | ~82% | ~0.80 |
| Naive Bayes | ~76% | ~0.73 |

### Regression (predicts: finishing position 1–20)
| Model | MAE | R² |
|---|---|---|
| Ridge Regression | ~2.7 | ~0.72 |
| Linear Regression | ~2.8 | ~0.71 |
| Lasso Regression | ~2.9 | ~0.70 |

### Unsupervised
- K-Means (k=5): Groups drivers into 5 performance tiers

---

## Features Used

| Feature | Source | Type |
|---|---|---|
| grid | results.csv | Raw |
| qual_position | qualifying.csv | Raw |
| year | races.csv | Raw |
| driverRef_enc | drivers.csv | Encoded |
| constructorRef_enc | constructors.csv | Encoded |
| circuitRef_enc | circuits.csv | Encoded |
| driver_avg_finish | Computed | Rolling-5 |
| team_avg_finish | Computed | Rolling-5 |
| driver_win_rate | Computed | Rolling-10 |
| pit_stop_count | pit_stops.csv | Aggregated |
| avg_lap_ms | lap_times.csv | Aggregated |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| SSL certificate error | Use `.\install.ps1` or add pip.ini |
| No module named joblib | SSL blocked install — use trusted-host flags |
| DVC duplicate output error | Already fixed in dvc.yaml — metrics.json only in evaluate stage |
| Models not found in app | Run `python src/train_models.py` first |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
| Docker exits immediately | Train models first so models/ folder exists |

---

## Future Scope

- MLflow visual experiment tracking
- Remote DVC storage (Google Drive / S3)
- Real-time Ergast API integration for live 2025 data
- Weather API integration for dynamic strategy adjustment
- LSTM neural network for lap-by-lap prediction
- Kubernetes deployment for production scale
