import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src", "ml"))

from split_and_baseline import (
    FEATURE_COLS,
    chronological_train_val_split,
    evaluate_predictions,
    generate_mock_telemetry,
)


def get_sample_df(tmp_path: str = "data/sample/test_mock_features.parquet") -> pd.DataFrame:
    if not os.path.exists(tmp_path):
        return generate_mock_telemetry(tmp_path)
    return pd.read_parquet(tmp_path)


def test_1_contract_a_schema_and_cloud_gaps():
    """Verifies Contract A columns exist and optical bands (NDVI/NDRE) preserve NaN cloud gaps."""
    df = get_sample_df()
    required_cols = set(FEATURE_COLS + [
        "grid_id", "lat", "lon", "date",
        "target_vci_t14", "target_vci_t28", "target_swdi_t14",
    ])
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing Contract A columns: {missing}"

    # Optical Sentinel-2 indices should have simulated cloud gaps (NaNs)
    assert df["ndvi"].isna().sum() > 0, "Expected NaN cloud gaps in 'ndvi'"
    assert df["ndre"].isna().sum() > 0, "Expected NaN cloud gaps in 'ndre'"

    # Gridded climate & static soil columns must have zero NaNs
    for col in ["spei_14", "spei_30", "gdd_cum", "vpd", "esi", "awc_0_30cm"]:
        assert df[col].isna().sum() == 0, f"Unexpected NaNs in {col}"


def test_2_strict_out_of_time_ordering():
    """Verifies train dates strictly precede validation dates with zero overlap."""
    df = get_sample_df()
    train_df, val_df = chronological_train_val_split(df, val_months=6, horizon_days=14)

    assert len(train_df) > 0 and len(val_df) > 0, "Train and validation splits must be non-empty"
    assert train_df["date"].max() < val_df["date"].min(), (
        f"Temporal leakage: max train date {train_df['date'].max()} >= min val date {val_df['date'].min()}"
    )


def test_3_purge_buffer_prevents_horizon_leakage():
    """Verifies at least `horizon_days` separates train_end_date and val_start_date."""
    df = get_sample_df()
    for horizon in (14, 28):
        train_df, val_df = chronological_train_val_split(df, val_months=6, horizon_days=horizon)
        gap_days = (val_df["date"].min() - train_df["date"].max()).days
        assert gap_days >= horizon, (
            f"Purge gap ({gap_days} days) is shorter than forecast horizon ({horizon} days)!"
        )


def test_4_metrics_logger_benchmarks_and_recall():
    """Verifies RMSE (<0.08), R2 (>0.82), and VCI < 35 Recall (>85%) calculations."""
    np.random.seed(0)
    y_true = np.linspace(15.0, 85.0, 200)
    # Simulate accurate predictions with small noise (~3.5 VCI points -> ~0.035 normalized RMSE)
    y_pred_q50 = y_true + np.random.normal(0, 3.5, size=len(y_true))
    # Lower 10th-percentile bound (q10) sits ~4 points lower, catching borderline VCI < 35 droughts
    y_pred_q10 = y_pred_q50 - 4.0

    metrics = evaluate_predictions(
        y_true_vci=y_true,
        y_pred_vci=y_pred_q50,
        drought_trigger_pred=y_pred_q10,
    )

    assert metrics["rmse_normalized"] < 0.08, f"Expected RMSE < 0.08, got {metrics['rmse_normalized']}"
    assert metrics["r2_score"] > 0.82, f"Expected R2 > 0.82, got {metrics['r2_score']}"
    assert metrics["drought_recall_vci35"] > 0.85, f"Expected Recall > 0.85, got {metrics['drought_recall_vci35']}"
    assert all([
        metrics["targets_met"]["rmse_below_0_08"],
        metrics["targets_met"]["r2_above_0_82"],
        metrics["targets_met"]["recall_above_0_85"],
    ])


def test_5_coordinate_lookup_latency_under_400ms():
    """Verifies single-coordinate lookup runs in < 400ms per Concept Note Section 6."""
    df = get_sample_df()
    _, val_df = chronological_train_val_split(df, val_months=6, horizon_days=14)

    metrics = evaluate_predictions(
        y_true_vci=val_df["target_vci_t14"].values,
        y_pred_vci=val_df["target_vci_t14"].values,
        zonal_lookup_df=val_df,
    )
    assert 0.0 < metrics["lookup_latency_ms"] < 400.0, (
        f"Latency {metrics['lookup_latency_ms']}ms exceeded the 400ms ceiling!"
    )


if __name__ == "__main__":
    for test_fn in [
        test_1_contract_a_schema_and_cloud_gaps,
        test_2_strict_out_of_time_ordering,
        test_3_purge_buffer_prevents_horizon_leakage,
        test_4_metrics_logger_benchmarks_and_recall,
        test_5_coordinate_lookup_latency_under_400ms,
    ]:
        test_fn()
        print(f"PASS: {test_fn.__name__}")
    print("\nAll 5 Day-1 validation & metrics tests passed!")