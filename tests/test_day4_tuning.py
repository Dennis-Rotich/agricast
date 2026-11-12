import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src", "ml"))


def test_day4_metrics_and_calibration():
    """Verifies that models/metrics.json contains tuned hyperparameters, meets targets, and calibrated recall >= 85%."""
    metrics_path = os.path.join(ROOT_DIR, "models", "metrics.json")
    assert os.path.exists(metrics_path), "models/metrics.json not found — run tune_optuna.py first"

    with open(metrics_path, "r") as f:
        data = json.load(f)

    # 1. Hyperparameters present
    hp = data.get("best_hyperparameters", {})
    for key in ("num_leaves", "max_depth", "learning_rate", "min_child_samples", "subsample", "colsample_bytree"):
        assert key in hp, f"Missing hyperparameter {key} in tuned metrics"

    # 2. Concept Note target verification
    m14 = data["vci_t14_eval"]
    assert m14["rmse_normalized"] < 0.08, f"RMSE {m14['rmse_normalized']} >= 0.08"
    assert m14["r2_score"] > 0.82, f"R² {m14['r2_score']} <= 0.82"

    # 3. Calibrated recall verification
    c14 = data["calibration_t14"]
    assert c14["drought_recall"] >= 0.85, f"Recall {c14['drought_recall']} < 0.85"
    assert c14["calibrated_trigger_threshold"] >= 30.0

    # 4. Latency verification
    assert m14["lookup_latency_ms"] < 400.0


def test_day4_joblib_model_horizons():
    """Verifies production .joblib contains all 4 targets and loads cleanly."""
    import joblib

    joblib_path = os.path.join(ROOT_DIR, "models", "lgbm_quantiles.joblib")
    assert os.path.exists(joblib_path)

    bundle = joblib.load(joblib_path)
    for target in ("target_vci_t14", "target_vci_t28", "target_swdi_t14", "target_swdi_t28"):
        assert target in bundle
        for q in ("q10", "q50", "q90"):
            assert q in bundle[target]