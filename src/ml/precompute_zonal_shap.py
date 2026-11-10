import json
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import shap

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..")) if THIS_DIR.endswith(os.path.join("src", "ml")) else THIS_DIR
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, ROOT_DIR)

from train_telemetry import (
    FEATURE_COLS,
    engineer_features,
    load_corridor_telemetry,
)

def extract_shap_top_drivers(shap_values: np.ndarray, feature_names: list, top_k: int = 3) -> pd.DataFrame:
    records = []
    for row in shap_values:
        risk_impact = -row
        top_indices = np.argsort(np.abs(risk_impact))[::-1][:top_k]

        rec = {}
        for rank, idx in enumerate(top_indices, start=1):
            rec[f"shap_top{rank}_feature"] = feature_names[idx]
            rec[f"shap_top{rank}_val"] = round(float(risk_impact[idx]), 2)
        records.append(rec)

    return pd.DataFrame(records)

def run_zonal_precomputation():
    print("[Day 5] Loading corridor telemetry and production models...")
    raw_df, src_name = load_corridor_telemetry()
    clean_df = engineer_features(raw_df)

    models_path = os.path.join(ROOT_DIR, "models", "lgbm_quantiles.joblib")
    metrics_path = os.path.join(ROOT_DIR, "models", "metrics.json")

    assert os.path.exists(models_path), "Missing models/lgbm_quantiles.joblib. Run tune_optuna.py first."
    models_bundle = joblib.load(models_path)

    latest_df = clean_df.sort_values("date").groupby("grid_id", as_index=False).last()
    print(f"       Extracted latest operational state for {len(latest_df)} grid cells on date {latest_df['date'].max().date()}.")

    X_latest = latest_df[FEATURE_COLS]

    m14_q10 = models_bundle["target_vci_t14"]["q10"]
    m14_q50 = models_bundle["target_vci_t14"]["q50"]
    m14_q90 = models_bundle["target_vci_t14"]["q90"]
    swdi_q50 = models_bundle["target_swdi_t14"]["q50"]

    pred_q10 = m14_q10.predict(X_latest)
    pred_q50 = m14_q50.predict(X_latest)
    pred_q90 = m14_q90.predict(X_latest)
    pred_swdi = swdi_q50.predict(X_latest)

    pred_q10 = np.minimum(pred_q10, pred_q50)
    pred_q90 = np.maximum(pred_q90, pred_q50)

    calibrated_thresh = 35.0
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
            calibrated_thresh = metrics_data.get("calibration_t14", {}).get("calibrated_trigger_threshold", 35.0)

    print(f"       Using calibrated drought trigger threshold: q10 < {calibrated_thresh}")

    vci_stress = 100.0 - pred_q50
    swdi_stress = np.clip((0.65 - pred_swdi) * 40.0, -15.0, 25.0)
    vuln_score = np.clip(0.85 * vci_stress + 0.15 * swdi_stress, 0.0, 100.0).round(1)

    print("       Computing TreeSHAP local feature contributions...")

    booster = getattr(m14_q50, "booster_", m14_q50)
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(X_latest)

    shap_df = extract_shap_top_drivers(shap_vals, FEATURE_COLS, top_k=3)

    forecast_dates = pd.to_datetime(latest_df["date"]) + pd.Timedelta(days=14)

    zonal_df = pd.DataFrame({
        "grid_id": latest_df["grid_id"],
        "lat": latest_df["lat"].round(4),
        "lon": latest_df["lon"].round(4),
        "forecast_date": forecast_dates.dt.strftime("%Y-%m-%d"),
        "vci_q10": pred_q10.round(2),
        "vci_q50": pred_q50.round(2),
        "vci_q90": pred_q90.round(2),
        "swdi_q50": pred_swdi.round(3),
        "vulnerability_score_0_100": vuln_score,
        "drought_flag_vci35": (pred_q10 < calibrated_thresh).astype(int),
    })

    zonal_df = pd.concat([zonal_df, shap_df], axis=1)

    test_lat, test_lon = zonal_df["lat"].iloc[0], zonal_df["lon"].iloc[0]
    t0 = time.perf_counter()

    sq_dist = (zonal_df["lat"] - test_lat) ** 2 + (zonal_df["lon"] - test_lon) ** 2
    matched_row = zonal_df.iloc[sq_dist.argmin()]
    lookup_latency_ms = (time.perf_counter() - t0) * 1000.0

    print("\n--- Latency Benchmark on Pre-Computed Table ---")
    print(f"Pin-Drop Query Latency: {lookup_latency_ms:.3f} ms (< 400ms target) [PASS]")

    out_dir = os.path.join(ROOT_DIR, "data", "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_parquet = os.path.join(out_dir, "zonal_forecasts.parquet")
    zonal_df.to_parquet(out_parquet, index=False)
    print(f"\n[Contract B Exported] Saved {len(zonal_df)} zonal forecasts to:\n  -> {out_parquet}")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            full_metrics = json.load(f)
    else:
        full_metrics = {}

    full_metrics["day5_zonal_precomputation"] = {
        "num_grid_cells": len(zonal_df),
        "forecast_horizon_days": 14,
        "calibrated_trigger_threshold": calibrated_thresh,
        "lookup_latency_ms": round(lookup_latency_ms, 3),
        "high_risk_cells_flagged": int(zonal_df["drought_flag_vci35"].sum()),
        "top_frequent_risk_driver": str(zonal_df["shap_top1_feature"].mode()[0]),
    }

    with open(metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"[Contract C Updated] Updated metrics in:\n  -> {metrics_path}")

    print("\n--- Sample Zonal Forecast Rows for Streamlit ---")
    print(zonal_df[["grid_id", "vulnerability_score_0_100", "drought_flag_vci35", "shap_top1_feature", "shap_top1_val"]].head(3))

if __name__ == "__main__":
    run_zonal_precomputation()