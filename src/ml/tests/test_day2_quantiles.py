import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

# Check both the root folder and src/ml/ so imports work regardless of layout
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src", "ml"))

from split_and_baseline import (
    chronological_train_val_split,
    generate_mock_telemetry,
)
from train_quantiles import (
    TARGETS,
    train_lgbm_quantiles,
    train_rf_baseline,
)

_CACHE = {}


def get_day2_fixture():
    """Trains RF baseline and LightGBM quantile models once and caches results for fast tests."""
    if "bundle" not in _CACHE:
        tmp_path = os.path.join(ROOT_DIR, "data", "sample", "test_mock_features.parquet")
        df = pd.read_parquet(tmp_path) if os.path.exists(tmp_path) else generate_mock_telemetry(tmp_path)
        train_df, val_df = chronological_train_val_split(df, val_months=6, horizon_days=14)

        rf_res = train_rf_baseline(train_df, val_df, "target_vci_t14")
        lgbm_models, lgbm_metrics, val_preds = train_lgbm_quantiles(train_df, val_df)

        _CACHE["bundle"] = (train_df, val_df, rf_res, lgbm_models, lgbm_metrics, val_preds)
    return _CACHE["bundle"]


def test_1_rf_baseline_handles_cloud_gaps():
    """Verifies Random Forest + median imputer trains cleanly on NaN cloud gaps."""
    _, val_df, rf_res, _, _, _ = get_day2_fixture()
    assert val_df["ndvi"].isna().sum() > 0, "Expected NaN cloud gaps in validation set"
    metrics = rf_res["metrics"]
    assert 0.0 < metrics["rmse_normalized"] < 0.15
    assert metrics["r2_score"] > 0.70
    assert 0.0 < metrics["lookup_latency_ms"] < 400.0


def test_2_lgbm_quantile_models_and_horizons():
    """Verifies LightGBM trains q10, q50, q90 across all 3 targets (t+14 VCI, t+28 VCI, t+14 SWDI)."""
    _, val_df, _, lgbm_models, _, val_preds = get_day2_fixture()

    for target_col in TARGETS:
        assert target_col in lgbm_models, f"Missing target {target_col} in trained models"
        for q_label in ("q10", "q50", "q90"):
            assert q_label in lgbm_models[target_col], f"Missing {q_label} for {target_col}"
            pred_col = f"{target_col}_{q_label}"
            assert pred_col in val_preds.columns, f"Missing prediction column {pred_col}"
            assert val_preds[pred_col].isna().sum() == 0, f"NaN predictions found in {pred_col}"
            assert len(val_preds[pred_col]) == len(val_df)


def test_3_quantile_monotonicity_interval():
    """Verifies 80% confidence interval ordering (q10 <= q50 <= q90) holds for every row."""
    _, _, _, _, _, val_preds = get_day2_fixture()

    for target_col in TARGETS:
        q10 = val_preds[f"{target_col}_q10"].values
        q50 = val_preds[f"{target_col}_q50"].values
        q90 = val_preds[f"{target_col}_q90"].values

        assert np.all(q10 <= q50 + 1e-6), f"Quantile crossing (q10 > q50) in {target_col}"
        assert np.all(q50 <= q90 + 1e-6), f"Quantile crossing (q50 > q90) in {target_col}"
        # Verify the 80% interval has positive width on average
        assert np.mean(q90 - q10) > 0.0, f"Zero-width confidence interval in {target_col}"


def test_4_vulnerability_score_and_benchmark_targets():
    """Verifies 0-100 Vulnerability Score bounds and Concept Note statistical benchmarks."""
    _, _, _, _, lgbm_metrics, val_preds = get_day2_fixture()

    scores = val_preds["vulnerability_score_0_100"].values
    assert np.all((scores >= 0.0) & (scores <= 100.0)), "Vulnerability Score out of [0, 100] bounds"

    m14 = lgbm_metrics["t14"]
    assert m14["rmse_normalized"] < 0.08, f"t+14 RMSE {m14['rmse_normalized']} >= 0.08"
    assert m14["r2_score"] > 0.82, f"t+14 R2 {m14['r2_score']} <= 0.82"
    assert m14["drought_recall_vci35"] > 0.85, f"t+14 Recall {m14['drought_recall_vci35']} <= 0.85"
    assert m14["lookup_latency_ms"] < 400.0, f"Latency {m14['lookup_latency_ms']}ms >= 400ms"


def test_5_contract_c_artifacts_serialization():
    """Verifies Contract C model bundle (.joblib) and metrics (.json) serialize and reload cleanly."""
    _, val_df, rf_res, lgbm_models, lgbm_metrics, _ = get_day2_fixture()

    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib_path = os.path.join(models_dir, "lgbm_quantiles.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")

    joblib.dump(lgbm_models, joblib_path)
    with open(metrics_path, "w") as f:
        json.dump({"rf_baseline_t14": rf_res["metrics"], "lgbm": lgbm_metrics}, f, indent=2)

    assert os.path.exists(joblib_path) and os.path.getsize(joblib_path) > 0
    assert os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0

    reloaded_models = joblib.load(joblib_path)
    sample_pred = reloaded_models["target_vci_t14"]["q50"].predict(val_df.iloc[:5][[
        "ndvi", "ndre", "spei_14", "spei_30", "gdd_cum", "vpd", "esi",
        "awc_0_30cm", "sand_clay_ratio", "soc_0_30cm",
    ]])
    assert len(sample_pred) == 5 and not np.any(np.isnan(sample_pred))


if __name__ == "__main__":
    for test_fn in [
        test_1_rf_baseline_handles_cloud_gaps,
        test_2_lgbm_quantile_models_and_horizons,
        test_3_quantile_monotonicity_interval,
        test_4_vulnerability_score_and_benchmark_targets,
        test_5_contract_c_artifacts_serialization,
    ]:
        test_fn()
        print(f"PASS: {test_fn.__name__}")
    print("\nAll 5 Day-2 Quantile & Baseline tests passed!")