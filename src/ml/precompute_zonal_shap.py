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

from split_and_baseline import chronological_train_val_split, evaluate_predictions
from train_telemetry import (
    FEATURE_COLS,
    engineer_features,
    load_corridor_telemetry,
)

def extract_shap_top_drivers(shap_values: np.ndarray, feature_names: list, top_k: int = 3) -> pd.DataFrame:
    risk_impact = -np.asarray(shap_values)
    top_idx = np.argsort(np.abs(risk_impact), axis=1)[:, ::-1][:, :top_k]
    feature_arr = np.asarray(feature_names)

    out = {}
    row_arange = np.arange(len(risk_impact))
    for rank in range(top_k):
        idx_col = top_idx[:, rank]
        out[f"shap_top{rank + 1}_feature"] = feature_arr[idx_col]
        out[f"shap_top{rank + 1}_val"] = np.round(risk_impact[row_arange, idx_col], 2)

    return pd.DataFrame(out)

def build_horizon_frame(
    val_df: pd.DataFrame,
    models_bundle: dict,
    horizon_days: int,
    calibrated_thresh: float,
) -> pd.DataFrame:
    X_val = val_df[FEATURE_COLS]
    vci_key = f"target_vci_t{horizon_days}"
    swdi_key = f"target_swdi_t{horizon_days}" if f"target_swdi_t{horizon_days}" in models_bundle else "target_swdi_t14"

    m_q10 = models_bundle[vci_key]["q10"]
    m_q50 = models_bundle[vci_key]["q50"]
    m_q90 = models_bundle[vci_key]["q90"]
    m_swdi = models_bundle[swdi_key]["q50"]

    pred_q10 = m_q10.predict(X_val)
    pred_q50 = m_q50.predict(X_val)
    pred_q90 = m_q90.predict(X_val)
    pred_swdi = m_swdi.predict(X_val)

    pred_q10 = np.minimum(pred_q10, pred_q50)
    pred_q90 = np.maximum(pred_q90, pred_q50)

    vci_stress = 100.0 - pred_q50
    swdi_stress = np.clip((0.65 - pred_swdi) * 40.0, -15.0, 25.0)
    vuln_score = np.clip(0.85 * vci_stress + 0.15 * swdi_stress, 0.0, 100.0).round(1)

    booster = getattr(m_q50, "booster_", m_q50)
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(X_val)
    shap_df = extract_shap_top_drivers(shap_vals, FEATURE_COLS, top_k=3)

    obs_dates = pd.to_datetime(val_df["date"])
    forecast_dates = obs_dates + pd.Timedelta(days=horizon_days)

    horizon_df = pd.DataFrame({
        "grid_id": val_df["grid_id"].values,
        "lat": val_df["lat"].round(4).values,
        "lon": val_df["lon"].round(4).values,
        "date": obs_dates.dt.strftime("%Y-%m-%d").values,
        "observation_date": obs_dates.dt.strftime("%Y-%m-%d").values,
        "forecast_date": forecast_dates.dt.strftime("%Y-%m-%d").values,
        "horizon_days": int(horizon_days),
        "vci_current": (val_df["ndvi"] * 100.0).round(2).values,
        "vci_actual": val_df[vci_key].round(2).values,
        "vci_q10": np.round(pred_q10, 2),
        "vci_q50": np.round(pred_q50, 2),
        "vci_q90": np.round(pred_q90, 2),
        "swdi_q50": np.round(pred_swdi, 3),
        "vulnerability_score_0_100": vuln_score,
        "drought_flag_vci35": (pred_q10 < calibrated_thresh).astype(int),
    })

    return pd.concat([horizon_df.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

def run_zonal_precomputation():
    print("[Day 5] Loading corridor telemetry and production models...")
    raw_df, src_name = load_corridor_telemetry()
    clean_df = engineer_features(raw_df)
    _, val_df = chronological_train_val_split(clean_df, val_months=6, horizon_days=14)

    models_path = os.path.join(ROOT_DIR, "models", "lgbm_quantiles.joblib")
    metrics_path = os.path.join(ROOT_DIR, "models", "metrics.json")

    assert os.path.exists(models_path), "Missing models/lgbm_quantiles.joblib. Run tune_optuna.py first."
    models_bundle = joblib.load(models_path)

    full_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            full_metrics = json.load(f)

    thresh_14 = full_metrics.get("calibration_t14", {}).get("calibrated_trigger_threshold", 35.0)
    thresh_28 = full_metrics.get("calibration_t28", {}).get("calibrated_trigger_threshold", 35.0)

    # 1. Build multi-date validation tables for BOTH horizon_days = 14 and horizon_days = 28
    print(f"       Pre-computing 14-day & 28-day forecasts + TreeSHAP across {len(val_df):,} validation rows...")
    df_14 = build_horizon_frame(val_df, models_bundle, horizon_days=14, calibrated_thresh=thresh_14)
    df_28 = build_horizon_frame(val_df, models_bundle, horizon_days=28, calibrated_thresh=thresh_28)

    zonal_df = pd.concat([df_14, df_28], ignore_index=True)

    # 2. Compute Persistence Baseline metrics (predicting VCI_{t+h} = VCI_t) for metrics.json
    vci_current = (val_df["ndvi"] * 100.0).values
    persistence_t14 = evaluate_predictions(
        y_true_vci=val_df["target_vci_t14"].values,
        y_pred_vci=vci_current,
        drought_trigger_pred=vci_current,
        zonal_lookup_df=val_df,
    )
    persistence_t28 = evaluate_predictions(
        y_true_vci=val_df["target_vci_t28"].values,
        y_pred_vci=vci_current,
        drought_trigger_pred=vci_current,
        zonal_lookup_df=val_df,
    )

    # 3. Benchmark Single-Coordinate + Horizon + Date Lookup Latency (< 400ms target)
    sample_date = df_14["date"].iloc[-1]
    sample_lat, sample_lon = df_14["lat"].iloc[0], df_14["lon"].iloc[0]
    t0 = time.perf_counter()
    slice_df = zonal_df[(zonal_df["horizon_days"] == 14) & (zonal_df["date"] == sample_date)]
    sq_dist = (slice_df["lat"] - sample_lat) ** 2 + (slice_df["lon"] - sample_lon) ** 2
    _ = slice_df.iloc[sq_dist.argmin()]
    lookup_latency_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Export Contract B Parquet
    out_dir = os.path.join(ROOT_DIR, "data", "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_parquet = os.path.join(out_dir, "zonal_forecasts.parquet")
    zonal_df.to_parquet(out_parquet, index=False)
    print(f"\n[Contract B Exported] Saved {len(zonal_df):,} rows (Horizons: {sorted(zonal_df['horizon_days'].unique().tolist())}, "
          f"Dates: {zonal_df['date'].nunique()}) to:\n  -> {out_parquet}")

    # 5. Add Persistence Baseline and Day 5 metadata to Contract C (models/metrics.json)
    persistence_block = {
        **persistence_t14,
        "t14": persistence_t14,
        "t28": persistence_t28,
        "vci_t14": persistence_t14,
        "vci_t28": persistence_t28,
    }
    full_metrics["persistence"] = persistence_block
    full_metrics["persistence_baseline"] = persistence_block
    full_metrics["persistence_t14"] = persistence_t14
    full_metrics["persistence_t28"] = persistence_t28

    full_metrics["day5_zonal_precomputation"] = {
        "num_total_rows": len(zonal_df),
        "num_grid_cells": int(zonal_df["grid_id"].nunique()),
        "num_validation_dates": int(zonal_df["date"].nunique()),
        "horizons_exported": [14, 28],
        "calibrated_trigger_threshold_t14": thresh_14,
        "calibrated_trigger_threshold_t28": thresh_28,
        "lookup_latency_ms": round(lookup_latency_ms, 3),
        "high_risk_rows_flagged": int(zonal_df["drought_flag_vci35"].sum()),
        "top_frequent_risk_driver": str(zonal_df["shap_top1_feature"].mode()[0]),
    }

    with open(metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"[Contract C Updated] Added persistence baseline & multi-horizon stats to:\n  -> {metrics_path}")
    print(f"Persistence Baseline (t+14) RMSE: {persistence_t14['rmse_normalized']} | R²: {persistence_t14['r2_score']}")
    print(f"Persistence Baseline (t+28) RMSE: {persistence_t28['rmse_normalized']} | R²: {persistence_t28['r2_score']}")


if __name__ == "__main__":
    run_zonal_precomputation()