"""
scripts/retrain.py
-------------------
Retraining strategy: checks for performance degradation or data drift,
then re-runs preprocessing + training if thresholds are exceeded.
Designed to be triggered by cron or CI/CD.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("retrain")

METRICS_PATH  = Path("models/metrics.json")
F1_THRESHOLD  = 0.85
DRIFT_THRESHOLD = 0.20   # fraction of features drifted


def load_current_metrics() -> dict:
    if not METRICS_PATH.exists():
        logger.warning("No metrics.json found — forcing retrain.")
        return {"test": {"f1": 0.0}}
    with open(METRICS_PATH) as f:
        return json.load(f)


def should_retrain(metrics: dict) -> tuple[bool, str]:
    """Return (should_retrain, reason)."""
    test_f1 = metrics.get("test", {}).get("f1", 0.0)
    if test_f1 < F1_THRESHOLD:
        return True, f"F1-Score {test_f1:.4f} below threshold {F1_THRESHOLD}"

    # Check drift report if available
    drift_path = Path("monitoring/reports/drift_latest.json")
    if drift_path.exists():
        with open(drift_path) as f:
            drift = json.load(f)
        frac = drift.get("summary", {}).get("drift_fraction", 0)
        if frac > DRIFT_THRESHOLD:
            return True, f"Drift fraction {frac:.2%} exceeds threshold {DRIFT_THRESHOLD:.2%}"

    return False, "All checks passed — no retraining needed."


def run_step(cmd: list[str], label: str) -> None:
    logger.info("Running: %s", label)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error("%s failed with code %d", label, result.returncode)
        sys.exit(result.returncode)
    logger.info("%s completed successfully.", label)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    metrics = load_current_metrics()
    retrain, reason = should_retrain(metrics)
    logger.info("Retrain decision: %s — %s", retrain, reason)

    if not retrain:
        print(f"No retraining required: {reason}")
        return

    print(f"Initiating retraining: {reason}")

    run_step([sys.executable, "src/data/preprocess.py"],   "Preprocessing")
    run_step([sys.executable, "src/models/train.py", "--n-trials", "30"], "Training")

    # Re-evaluate
    new_metrics = load_current_metrics()
    new_f1 = new_metrics.get("test", {}).get("f1", 0)
    logger.info("Post-retrain F1-Score: %.6f", new_f1)
    print(f"Retraining complete — new F1-Score: {new_f1:.6f}")


if __name__ == "__main__":
    main()
