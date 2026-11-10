import json
import os
import sys
import pandas as pd

ROOT_DIR = "/home/dennis/agricast"

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src", "ml"))

def test_contract_b_schema_and_monotonicity():
    """Verifies that Contract B parquet exists, matches required columns, and satisfies quantile monotonicity."""
    parquet_path = os.path.join(ROOT_DIR, "data", "predictions", "zonal_forecasts.parquet")
    assert os.path.exists(parquet_path), "data/predictions/zonal_forecasts.parquet missing"

    df = pd.read_parquet(parquet_path)
    required_cols = [
        "grid_id", "lat", "lon", "forecast_date",
        "vci_q10", "vci_q50", "vci_q90", "swdi_q50",
        "vulnerability_score_0_100", "drought_flag_vci35",
        "shap_top1_feature", "shap_top1_val",
        "shap_top2_feature", "shap_top2_val",
        "shap_top3_feature", "shap_top3_val",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing Contract B column: {col}"

    assert len(df) > 0, "Zonal forecasts table is empty"
    assert (df["vci_q10"] <= df["vci_q50"]).all(), "Quantile crossing: q10 > q50"
    assert (df["vci_q50"] <= df["vci_q90"]).all(), "Quantile crossing: q50 > q90"
    assert ((df["vulnerability_score_0_100"] >= 0.0) & (df["vulnerability_score_0_100"] <= 100.0)).all()


def test_contract_c_metrics_and_sub_400ms_latency():
    """Verifies metrics.json records Day 5 zonal verification and < 400ms latency."""
    metrics_path = os.path.join(ROOT_DIR, "models", "metrics.json")
    assert os.path.exists(metrics_path)

    with open(metrics_path, "r") as f:
        data = json.load(f)

    d5 = data.get("day5_zonal_precomputation", {})
    assert d5.get("num_grid_cells", 0) > 0
    assert d5.get("lookup_latency_ms", 999.0) < 400.0, "Lookup latency exceeded 400ms"