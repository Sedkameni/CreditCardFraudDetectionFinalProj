"""tests/unit/test_preprocess.py"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import eda_summary, preprocess, split_and_resample


@pytest.fixture
def sample_df():
    """Create a small synthetic dataframe mimicking creditcard.csv."""
    rng = np.random.default_rng(0)
    n = 500
    data = {col: rng.normal(size=n) for col in [f"V{i}" for i in range(1, 29)]}
    data["Amount"] = rng.exponential(scale=100, size=n)
    data["Time"]   = np.arange(n, dtype=float)
    # Heavily imbalanced: ~1% fraud
    data["Class"]  = (rng.random(n) < 0.01).astype(int)
    return pd.DataFrame(data)


def test_eda_summary_counts(sample_df):
    summary = eda_summary(sample_df)
    assert summary["total_rows"] == 500
    assert summary["missing_values"] == 0
    assert summary["fraud_count"] + summary["legitimate_count"] == 500


def test_preprocess_drops_time_and_amount(sample_df):
    X, y = preprocess(sample_df.copy())
    assert "Time"   not in X.columns
    assert "Amount" not in X.columns
    assert "Amount_scaled" in X.columns
    assert "Class" not in X.columns
    assert len(y) == len(X)


def test_preprocess_no_missing(sample_df):
    X, y = preprocess(sample_df.copy())
    assert X.isnull().sum().sum() == 0


def test_split_sizes(sample_df):
    X, y = preprocess(sample_df.copy())
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_resample(
        X, y, use_smote=False
    )
    total_test_val = len(X_val) + len(X_test)
    # test ~15%, val ~15%
    assert total_test_val > 0
    assert len(X_train) > len(X_val)


def test_smote_balances_classes(sample_df):
    X, y = preprocess(sample_df.copy())
    if y.sum() < 6:
        pytest.skip("Not enough fraud samples for SMOTE k=5")
    X_train, _, _, y_train, _, _ = split_and_resample(X, y, use_smote=True)
    # After SMOTE the minority class should be much larger
    fraud_ratio = y_train.sum() / len(y_train)
    assert fraud_ratio > 0.3, f"Expected > 30% fraud after SMOTE, got {fraud_ratio:.2%}"
