"""
src/models/train.py
--------------------
XGBoost training with Optuna hyperparameter optimisation,
MLflow experiment tracking, and artefact persistence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "fraud-detection-xgboost"
N_TRIALS = 50
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_split(name: str, dir_: Path = PROCESSED_DIR) -> np.ndarray:
    df = pd.read_parquet(dir_ / f"{name}.parquet")
    return df.values.ravel() if df.shape[1] == 1 else df.values


def load_all_splits() -> Tuple:
    X_train = load_split("X_train")
    X_val   = load_split("X_val")
    X_test  = load_split("X_test")
    y_train = load_split("y_train")
    y_val   = load_split("y_val")
    y_test  = load_split("y_test")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray) -> Dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return {
        "roc_auc":          round(roc_auc_score(y, y_prob), 6),
        "avg_precision":    round(average_precision_score(y, y_prob), 6),
        "f1":               round(f1_score(y, y_pred), 6),
        "precision":        round(precision_score(y, y_pred), 6),
        "recall":           round(recall_score(y, y_pred), 6),
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def build_objective(X_train, y_train, X_val, y_val):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 200, 1000, step=50),
            "max_depth":          trial.suggest_int("max_depth", 3, 10),
            "learning_rate":      trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "gamma":              trial.suggest_float("gamma", 0, 5),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "scale_pos_weight":   trial.suggest_float("scale_pos_weight", 1, 500),
            "use_label_encoder":  False,
            "eval_metric":        "aucpr",
            "random_state":       RANDOM_STATE,
            "n_jobs":             -1,
            "tree_method":        "hist",
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        metrics = evaluate(model, X_val, y_val)
        return metrics["avg_precision"]   # optimise PR-AUC

    return objective


# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------

def train(
    n_trials: int = N_TRIALS,
    use_mlflow: bool = True,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> xgb.XGBClassifier:

    logger.info("Loading processed data splits…")
    X_train, X_val, X_test, y_train, y_val, y_test = load_all_splits()
    logger.info("Train: %s | Val: %s | Test: %s", X_train.shape, X_val.shape, X_test.shape)

    # ---------- Optuna HPO ----------
    logger.info("Starting Optuna HPO — %d trials…", n_trials)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(),
        study_name="xgb-fraud",
    )
    study.optimize(
        build_objective(X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    logger.info("Best params: %s", json.dumps(best_params, indent=2))

    # ---------- Final model ----------
    logger.info("Training final model with best params…")
    best_params.update({
        "use_label_encoder": False,
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    })

    model = xgb.XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ---------- Evaluation ----------
    val_metrics  = evaluate(model, X_val, y_val)
    test_metrics = evaluate(model, X_test, y_test)

    logger.info("=== Validation Metrics ===")
    for k, v in val_metrics.items():
        logger.info("  %-20s: %s", k, v)

    logger.info("=== Test Metrics ===")
    for k, v in test_metrics.items():
        logger.info("  %-20s: %s", k, v)

    y_test_pred = model.predict(X_test)
    logger.info("\n%s", classification_report(y_test, y_test_pred, target_names=["Legit", "Fraud"]))

    # ---------- MLflow ----------
    if use_mlflow:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name="xgb-best"):
            mlflow.log_params(best_params)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            mlflow.xgboost.log_model(model, artifact_path="model")
            logger.info("Logged to MLflow experiment: %s", EXPERIMENT_NAME)

    # ---------- Persist ----------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "xgb_fraud_model.joblib"
    joblib.dump(model, model_path)
    logger.info("Model saved → %s", model_path)

    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics, "best_params": best_params}, f, indent=2)
    logger.info("Metrics saved → %s", metrics_path)

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train XGBoost fraud detection model")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS, help="Optuna HPO trials")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    args = parser.parse_args()

    train(n_trials=args.n_trials, use_mlflow=not args.no_mlflow)
