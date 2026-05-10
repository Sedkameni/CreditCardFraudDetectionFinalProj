# Pipeline Architecture

## Overview

This document describes the end-to-end MLOps architecture for the Credit Card Fraud Detection system.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Ingestion                            │
│  Kaggle CSV  ──►  data/raw/creditcard.csv                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Preprocessing Pipeline                         │
│  src/data/preprocess.py                                          │
│  • Deduplication        • Missing-value imputation               │
│  • RobustScaler(Amount) • Train / Val / Test split               │
│  • SMOTE oversampling   • Parquet output                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Model Training                             │
│  src/models/train.py                                             │
│  • XGBoost XGBClassifier                                         │
│  • Optuna TPE HPO (50 trials, optimise PR-AUC)                   │
│  • MLflow experiment tracking                                    │
│  • Artefacts: models/xgb_fraud_model.joblib                      │
│               models/metrics.json                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CI / CD (GitHub Actions)                        │
│  .github/workflows/ci.yml   .github/workflows/cd.yml            │
│  • Lint → Test → Docker build → Staging deploy                   │
│  • Weekly scheduled retraining via scripts/retrain.py            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                          │
│                                                                  │
│   ┌─────────────┐   ┌────────────┐   ┌──────────┐              │
│   │  FastAPI    │   │ Prometheus │   │ Grafana  │              │
│   │  :8000      │──►│  :9090     │──►│  :3000   │              │
│   └─────────────┘   └────────────┘   └──────────┘              │
│          │                                                       │
│   ┌─────────────┐                                               │
│   │   MLflow    │                                               │
│   │   :5000     │                                               │
│   └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Drift Monitoring                             │
│  src/monitoring/drift.py                                         │
│  • KS-test per feature                                           │
│  • F1-Score degradation alert (< 0.85)                           │
│  • JSON drift report → monitoring/reports/                       │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1 — Data Layer
- **Raw data**: `data/raw/creditcard.csv` (284,807 rows, 31 cols)
- **Processed**: Parquet files in `data/processed/` (train / val / test splits)
- **Features**: V1–V28 (PCA-transformed), Amount_scaled

### 2 — Preprocessing
| Step | Detail |
|---|---|
| Deduplication | 1,081 duplicates removed |
| Missing values | None in source; imputation guard with column median |
| Feature scaling | `RobustScaler` on Amount (resistant to outliers) |
| Class imbalance | SMOTE k=5, applied to train split only |

### 3 — Model
| Parameter | Value |
|---|---|
| Algorithm | XGBoost `XGBClassifier` |
| Objective | `binary:logistic` |
| HPO | Optuna TPE, 50 trials |
| Optimisation metric | PR-AUC (Average Precision) |
| Expected test F1 | ~0.87 |
| Expected ROC-AUC | ~0.98 |

### 4 — API
- Framework: **FastAPI 0.111**
- Serialisation: **Pydantic v2**
- Workers: 2× Uvicorn
- Endpoints: `/health`, `/predict`, `/predict/batch`, `/metrics`, `/model/info`

### 5 — CI/CD
```
Push to develop → CI (lint + test + docker build)
Push to main    → CI → CD (push GHCR image → staging deploy)
Cron (Sunday)   → Retrain → Commit new model → Deploy
```

### 6 — Monitoring
- **Prometheus**: scrapes `/metrics` every 10 s
- **Grafana**: pre-built dashboard (`docs/grafana_dashboard.json`)
- **Drift detection**: KS-test weekly; triggers retraining if >20% features drift

## Retraining Triggers

| Trigger | Condition | Action |
|---|---|---|
| Scheduled | Every Sunday 02:00 UTC | Run full retrain pipeline |
| Performance drift | Test F1 < 0.85 | Retrain + redeploy |
| Feature drift | >20% features with KS p < 0.05 | Retrain + redeploy |
| Manual | `python scripts/retrain.py` | Retrain on demand |

## Security Considerations
- API runs as non-root user inside container
- Model artefacts mounted read-only in API container
- Secrets managed via GitHub Actions encrypted secrets
- Grafana admin credentials set via environment variables (rotate in production)
