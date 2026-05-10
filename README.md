# Credit Card Fraud Detection — MLOps Pipeline

A production-ready MLOps pipeline for binary classification of credit card fraud using XGBoost, FastAPI, Docker, and GitHub Actions.

---

## Project Overview

| Item | Detail |
|---|---|
| **Dataset** | UCI Credit Card Fraud Detection (Kaggle) |
| **Model** | XGBoost (binary classifier) |
| **API** | FastAPI (REST) |
| **Containerisation** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Monitoring** | Prometheus + Grafana |

---

## Repository Structure

```
mlops-project/
├── src/
│   ├── data/           # Data loading, EDA, preprocessing
│   ├── models/         # Training, evaluation, hyperparameter tuning
│   ├── api/            # FastAPI application & prediction logic
│   └── monitoring/     # Metrics collection & drift detection
├── tests/              # Unit & integration tests
├── notebooks/          # EDA & experiment Jupyter notebooks
├── .github/workflows/  # CI/CD pipeline definitions
├── config/             # Hyperparameter & app configuration
├── scripts/            # Utility scripts (retrain, evaluate)
├── docs/               # Technical documentation
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### Prerequisites
- Docker ≥ 24 & Docker Compose ≥ 2
- Python 3.11+
- 8 GB RAM recommended

### 1 — Clone & configure
```bash
git clone https://github.com/<your-org>/mlops-fraud-detection.git
cd mlops-fraud-detection
cp config/config.example.yaml config/config.yaml
```

### 2 — Add the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:
```
data/raw/creditcard.csv
```

### 3 — Train the model
```bash
# Option A — local Python environment
pip install -r requirements.txt
python src/models/train.py

# Option B — Docker
docker compose run trainer
```

### 4 — Start all services
```bash
docker compose up -d
```

Services:
| Service | URL |
|---|---|
| Prediction API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## API Usage

### Health check
```bash
curl http://localhost:8000/health
```

### Single prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
      "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
      "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
      "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
      "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
      "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
      "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
      "Amount": 149.62
    }
  }'
```

### Batch prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [...]}'
```

---

## CI/CD

Every push to `main` or `develop`:
1. Linting (`flake8`, `black`)
2. Unit tests (`pytest`)
3. Model integration tests
4. Docker image build
5. Auto-deploy to staging (on `main`)

---

## Model Retraining

Automated retraining is triggered by:
- **Schedule**: Weekly cron job (Sundays 02:00 UTC)
- **Performance drift**: F1-Score drops below 0.85 threshold
- **Manual**: `python scripts/retrain.py`

---

## Monitoring

- **Prediction latency** — p50/p95/p99 histograms
- **Fraud rate** — rolling 1 h / 24 h windows
- **Model drift** — feature distribution via KS-test
- **Data quality** — missing-value & range alerts

Import `docs/grafana_dashboard.json` into Grafana to activate the pre-built dashboard.

---

## Documentation

| Document | Location |
|---|---|
| Pipeline Architecture | `docs/architecture.md` |
| API Reference | `docs/api_reference.md` |
| Retraining Guide | `docs/retraining.md` |
| Final Report (PDF) | `docs/final_report.pdf` |

---

## License

MIT © 2025
