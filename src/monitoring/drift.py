"""
src/monitoring/drift.py
------------------------
Feature-distribution drift detection using the Kolmogorov-Smirnov test.
Intended to run on a schedule or triggered by prediction volume.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("drift_monitor")

FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled"]
DRIFT_REPORT_DIR = Path("monitoring/reports")
KS_P_VALUE_THRESHOLD = 0.05        # below this → drift detected
DRIFT_F1_THRESHOLD   = 0.85        # below this → trigger retraining


# ---------------------------------------------------------------------------
# KS-based drift detection
# ---------------------------------------------------------------------------

def compute_ks_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Compute Kolmogorov-Smirnov statistic for each feature.
    Returns per-feature dict with ks_stat, p_value, and drift_detected flag.
    """
    features = features or FEATURE_NAMES
    results: Dict[str, Dict] = {}

    for feat in features:
        if feat not in reference.columns or feat not in current.columns:
            continue
        ks_stat, p_value = stats.ks_2samp(
            reference[feat].dropna().values,
            current[feat].dropna().values,
        )
        results[feat] = {
            "ks_stat": round(float(ks_stat), 6),
            "p_value": round(float(p_value), 6),
            "drift_detected": bool(p_value < KS_P_VALUE_THRESHOLD),
        }

    drifted = [f for f, v in results.items() if v["drift_detected"]]
    drift_fraction = len(drifted) / len(results) if results else 0
    logger.info(
        "Drift check: %d/%d features drifted (%.1f%%)",
        len(drifted), len(results), drift_fraction * 100,
    )
    return results


def drift_summary(ks_results: Dict[str, Dict]) -> Dict:
    drifted = [f for f, v in ks_results.items() if v["drift_detected"]]
    return {
        "total_features_checked": len(ks_results),
        "features_with_drift":    len(drifted),
        "drift_fraction":         round(len(drifted) / max(len(ks_results), 1), 4),
        "drifted_features":       drifted,
        "requires_retraining":    len(drifted) / max(len(ks_results), 1) > 0.2,
    }


# ---------------------------------------------------------------------------
# Performance degradation check
# ---------------------------------------------------------------------------

def check_performance_degradation(current_f1: float) -> Dict:
    return {
        "current_f1": round(current_f1, 6),
        "threshold":  DRIFT_F1_THRESHOLD,
        "degraded":   current_f1 < DRIFT_F1_THRESHOLD,
        "action":     "retrain" if current_f1 < DRIFT_F1_THRESHOLD else "none",
    }


# ---------------------------------------------------------------------------
# Persist report
# ---------------------------------------------------------------------------

def save_drift_report(
    ks_results: Dict,
    perf_check: Optional[Dict] = None,
    tag: str = "latest",
) -> Path:
    DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "tag": tag,
        "ks_drift": ks_results,
        "summary": drift_summary(ks_results),
    }
    if perf_check:
        report["performance"] = perf_check

    out = DRIFT_REPORT_DIR / f"drift_{tag}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Drift report saved → %s", out)
    return out


# ---------------------------------------------------------------------------
# CLI / scheduler entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    ref_path = Path("data/processed/X_train.parquet")
    cur_path = Path("data/processed/X_test.parquet")

    if not ref_path.exists() or not cur_path.exists():
        raise SystemExit("Processed data not found — run preprocess.py first.")

    reference = pd.read_parquet(ref_path)
    current   = pd.read_parquet(cur_path)

    ks_results = compute_ks_drift(reference, current)
    summary    = drift_summary(ks_results)

    print("\n=== Drift Summary ===")
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")

    save_drift_report(ks_results, tag="manual_run")
