"""
src/data/preprocess.py
-----------------------
EDA helpers, preprocessing pipeline, and SMOTE oversampling
for the Credit Card Fraud Detection dataset.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Silence the joblib/loky warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DATA_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DIR = Path("data/processed")
TARGET_COL = "Class"
AMOUNT_COL = "Amount"
TIME_COL = "Time"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------

def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and validate expected columns."""
    logger.info("Loading raw data from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
        
    df = pd.read_csv(path)
    expected_cols = {TIME_COL, AMOUNT_COL, TARGET_COL}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])
    return df


def eda_summary(df: pd.DataFrame) -> dict:
    """Return a dict with key EDA statistics."""
    fraud_count = df[TARGET_COL].sum()
    total = len(df)
    fraud_pct = (fraud_count / total) * 100

    summary = {
        "total_rows": total,
        "total_columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "fraud_count": int(fraud_count),
        "legitimate_count": int(total - fraud_count),
        "fraud_percentage": round(fraud_pct, 4),
        "amount_mean": round(df[AMOUNT_COL].mean(), 2),
        "amount_std": round(df[AMOUNT_COL].std(), 2),
        "amount_max": round(df[AMOUNT_COL].max(), 2),
    }

    logger.info(
        "EDA: %d total | %d fraud (%.4f%%) | %d missing",
        total, fraud_count, fraud_pct, summary["missing_values"]
    )
    return summary


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline:
    1. Drop duplicates (explicit copy to avoid SettingWithCopyWarning)
    2. Handle missing values
    3. Scale `Amount` with RobustScaler
    4. Drop `Time` and raw `Amount`
    """
    logger.info("Starting preprocessing…")

    # 1. Duplicates - .copy() ensures we aren't working on a slice view
    before = len(df)
    df = df.drop_duplicates().copy()
    logger.info("Removed %d duplicates", before - len(df))

    # 2. Missing values — impute with median
    if df.isnull().sum().sum() > 0:
        logger.warning("Found missing values — filling with column median")
        df = df.fillna(df.median(numeric_only=True))

    # 3. Scale Amount using .loc to prevent warnings
    scaler = RobustScaler()
    df.loc[:, "Amount_scaled"] = scaler.fit_transform(df[[AMOUNT_COL]])

    # 4. Final Feature Selection
    # Dropping Time and original Amount; keeping PCA components (V1-V28)
    X = df.drop(columns=[TARGET_COL, TIME_COL, AMOUNT_COL], errors="ignore")
    y = df[TARGET_COL]

    logger.info("Preprocessing complete: X shape=%s, fraud=%d", X.shape, y.sum())
    return X, y


# ---------------------------------------------------------------------------
# Train / validation / test split + SMOTE
# ---------------------------------------------------------------------------

def split_and_resample(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    use_smote: bool = True,
) -> Tuple:
    """
    Split into train / val / test sets.
    Apply SMOTE only to the training split to prevent data leakage.
    """
    # First hold out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Then validation from the remaining
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    logger.info(
        "Split — train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    if use_smote:
        logger.info("Applying SMOTE to training set…")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(
            "After SMOTE — train: %d | fraud: %d | legit: %d",
            len(X_train), y_train.sum(), (y_train == 0).sum(),
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Persist processed artifacts
# ---------------------------------------------------------------------------

def save_processed(
    X_train, X_val, X_test, y_train, y_val, y_test,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    """Save datasets as Parquet files for efficient loading."""
    out_dir.mkdir(parents=True, exist_ok=True)

    data_map = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }

    for name, data in data_map.items():
        path = out_dir / f"{name}.parquet"
        # Ensure y (Series) is saved as a DataFrame for Parquet compatibility
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        data.to_parquet(path, index=False)
        logger.info("Saved %s → %s", name, path)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        # 1. Load & EDA
        raw_df = load_raw()
        stats = eda_summary(raw_df)
        
        print("\n=== EDA Summary ===")
        for k, v in stats.items():
            print(f"  {k:25s}: {v}")

        # 2. Preprocess
        X_clean, y_clean = preprocess(raw_df)

        # 3. Split & Oversample
        sets = split_and_resample(X_clean, y_clean)

        # 4. Save
        save_processed(*sets)
        
        print("\nPipeline execution successful.")

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        raise