import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except OSError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_LGBM = False

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

from split_and_baseline import (
    FEATURE_COLS as BASE_FEATURE_COLS,
    chronological_train_val_split,
    evaluate_predictions,
    generate_mock_telemetry
)

NEW_COLS = [
    "ndvi_lag7",
    "ndre_lag7",
    "delta_ndre_14d",
    "spei_30_lag7",
    "spei30_x_awc",
]

FEATURE_COLS = BASE_FEATURE_COLS + NEW_COLS
TARGETS = [
    "target_vci_t14",
    "target_vci_t28",
    "target_swdi_t14",
    "target_swdi_t28",
]
QUANTILES = [0.1,0.5,0.9]

def load_corridor_telemetry() -> tuple[pd.DataFrame, str]:
    real_path = os.path.join(ROOT_DIR, "data", "processed", "pilot_features.parquet")
    mock_path = os.path.join(ROOT_DIR, "data", "sample", "mock_features.parquet")

    if os.path.exists(real_path):
        df = pd.read_parquet(real_path)
        source = "real_corridor_telemetry"
    else:
        df = pd.read_parquet(mock_path) if os.path.exists(mock_path) else generate_mock_telemetry(mock_path)
        source = "mock_corridor_telemetry"

    if "target_swdi_t28" not in df.columns:
        df["target_swdi_t28"] = np.clip((df["target_vci_t28"] - 50.0) / 15.0, -3.5, 2.5)

    return df, source

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aligns 5-day Sentinel-2 passes with daily weather, engineers lags/interactions, and drops NaN targets."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["grid_id", "date"]).reset_index(drop=True)
    grp = df.groupby("grid_id", sort=False)

    # Detect whether dataset is daily (1-day step, like real pilot_features.parquet) or 5-day mock data
    sample_dates = df[df["grid_id"] == df["grid_id"].iloc[0]]["date"]
    median_step_days = int(max(1, sample_dates.diff().dt.days.median() or 1))

    if median_step_days == 1:
        # 1. Mask unmasked cloud-shadow drops (NDVI < 0.20) in real Sentinel-2 observations
        ndvi_clean = df["ndvi"].where(df["ndvi"] >= 0.20, np.nan)
        ndre_clean = df["ndre"].where(df["ndvi"] >= 0.20, np.nan)

        # 2. Interpolate 5-day satellite passes onto daily grid & apply causal 30d phenology smoothing
        df["ndvi"] = ndvi_clean.groupby(df["grid_id"]).transform(
            lambda s: s.interpolate(method="linear").ffill().bfill().rolling(30, min_periods=1).mean().rolling(30, min_periods=1).mean()
        )
        df["ndre"] = ndre_clean.groupby(df["grid_id"]).transform(
            lambda s: s.interpolate(method="linear").ffill().bfill().rolling(30, min_periods=1).mean().rolling(30, min_periods=1).mean()
        )

        # 3. Normalize NDVI per grid to [0, 1] so optical features match each grid's VCI (0-100) baseline
        g_min = df.groupby("grid_id")["ndvi"].transform("min")
        g_max = df.groupby("grid_id")["ndvi"].transform("max")
        df["ndvi"] = (df["ndvi"] - g_min) / (g_max - g_min + 1e-6)
        vci_daily = df["ndvi"] * 100.0

        # 4. Recompute aligned t+14 and t+28 VCI & SWDI targets so y has zero middle NaNs
        df["target_vci_t14"] = vci_daily.groupby(df["grid_id"]).shift(-14)
        df["target_vci_t28"] = vci_daily.groupby(df["grid_id"]).shift(-28)
        df["target_swdi_t14"] = grp["target_swdi_t14"].transform(lambda s: s.ffill().bfill())
        df["target_swdi_t28"] = df.groupby("grid_id")["target_swdi_t14"].shift(-14)

        lag_7, lag_14 = 7, 14
    else:
        if "target_swdi_t28" not in df.columns:
            df["target_swdi_t28"] = np.clip((df["target_vci_t28"] - 50.0) / 15.0, -3.5, 2.5)
        lag_7, lag_14 = 1, 2

    # 5. Compute per-grid autoregressive lags and biophysical soil-moisture interaction
    df["ndvi_lag7"] = grp["ndvi"].shift(lag_7)
    df["ndre_lag7"] = grp["ndre"].shift(lag_7)
    df["delta_ndre_14d"] = df["ndre"] - grp["ndre"].shift(lag_14)
    df["spei_30_lag7"] = grp["spei_30"].shift(lag_7)
    df["spei30_x_awc"] = df["spei_30"] * (df["awc_0_30cm"] / 100.0)

    # 6. Drop warmup/tail rows where any target y or lag is NaN (prevents ValueError: Input y contains NaN)
    required_non_null = TARGETS + ["spei_30_lag7", "delta_ndre_14d"]
    df_clean = df.dropna(subset=required_non_null).copy()

    # Subsample daily grid to 5-day Sentinel-2 cadence (~20k rows) for fast, responsive training
    if median_step_days == 1:
        df_clean["step_idx"] = df_clean.groupby("grid_id").cumcount()
        df_clean = df_clean[df_clean["step_idx"] % 5 == 0].drop(columns=["step_idx"])

    return df_clean.sort_values(["date", "grid_id"]).reset_index(drop=True)

def build_quantile_estimator(alpha: float):
    if HAS_LGBM:
        return LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=400,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=1,
        )
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=alpha,
        max_iter=400,
        learning_rate=0.035,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )

def run_training_and_comparison(df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    df_feat = engineer_features(df)
    train_df, val_df = chronological_train_val_split(df_feat, val_months=6, horizon_days=14)

    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)),
    ])
    rf_pipe.fit(train_df[FEATURE_COLS], train_df["target_vci_t14"])
    rf_preds = rf_pipe.predict(val_df[FEATURE_COLS])
    rf_metrics = evaluate_predictions(val_df["target_vci_t14"].values, rf_preds, zonal_lookup_df=val_df)

    base_q50 = build_quantile_estimator(0.5)
    base_q50.fit(train_df[BASE_FEATURE_COLS], train_df["target_vci_t14"])
    base_preds = base_q50.predict(val_df[BASE_FEATURE_COLS])
    base_metrics = evaluate_predictions(val_df["target_vci_t14"].values, base_preds, zonal_lookup_df=val_df)

    models = {}
    val_preds = val_df[["grid_id", "lat", "lon", "date"]].copy()

    for target_col in TARGETS:
        models[target_col] = {}
        for alpha in QUANTILES:
            q_label = f"q{int(alpha * 100)}"
            est = build_quantile_estimator(alpha)
            est.fit(train_df[FEATURE_COLS], train_df[target_col])
            models[target_col][q_label] = est
            val_preds[f"{target_col}_{q_label}"] = est.predict(val_df[FEATURE_COLS])

        q10 = val_preds[f"{target_col}_q10"].values
        q50 = val_preds[f"{target_col}_q50"].values
        q90 = val_preds[f"{target_col}_q90"].values
        val_preds[f"{target_col}_q10"] = np.minimum(q10, q50)
        val_preds[f"{target_col}_q90"] = np.maximum(q90, q50)

    vci_stress = 100.0 - val_preds["target_vci_t14_q50"]
    swdi_stress = np.clip(-val_preds["target_swdi_t14_q50"] * 12.0, -15.0, 25.0)
    val_preds["vulnerability_score_0_100"] = np.clip(0.85 * vci_stress + 0.15 * swdi_stress, 0.0, 100.0).round(1)

    metrics = {
        "vci_t14": evaluate_predictions(
            val_df["target_vci_t14"].values,
            val_preds["target_vci_t14_q50"].values,
            drought_trigger_pred=val_preds["target_vci_t14_q10"].values,
            zonal_lookup_df=val_preds,
        ),
        "vci_t28": evaluate_predictions(
            val_df["target_vci_t28"].values,
            val_preds["target_vci_t28_q50"].values,
            drought_trigger_pred=val_preds["target_vci_t28_q10"].values,
            zonal_lookup_df=val_preds,
        ),
        "swdi_t14_rmse": round(float(np.sqrt(np.mean((val_df["target_swdi_t14"] - val_preds["target_swdi_t14_q50"]) ** 2))), 4),
        "swdi_t28_rmse": round(float(np.sqrt(np.mean((val_df["target_swdi_t28"] - val_preds["target_swdi_t28_q50"]) ** 2))), 4),
    }

    comparison = {
        "rf_baseline_vci_t14": rf_metrics,
        "lgbm_base_10_features_vci_t14": base_metrics,
        "lgbm_day3_15_features": metrics,
        "feature_columns": FEATURE_COLS,
    }
    return models, comparison, val_preds

if __name__ == "__main__":
    df, src_name = load_corridor_telemetry()
    print(f"[Day 3] Loaded {len(df):,} rows from {src_name}")

    models, comparison, val_preds = run_training_and_comparison(df)
    print("\n--- Day 3 Model Comparison Summary ---")
    print("1. RF Baseline (t+14 VCI) RMSE:", comparison["rf_baseline_vci_t14"]["rmse_normalized"],
          "| R2:", comparison["rf_baseline_vci_t14"]["r2_score"])
    print("2. LightGBM Base 10 Features (t+14 VCI) RMSE:", comparison["lgbm_base_10_features_vci_t14"]["rmse_normalized"],
          "| R2:", comparison["lgbm_base_10_features_vci_t14"]["r2_score"])
    print("3. LightGBM Day-3 15 Features (t+14 VCI) RMSE:", comparison["lgbm_day3_15_features"]["vci_t14"]["rmse_normalized"],
          "| R2:", comparison["lgbm_day3_15_features"]["vci_t14"]["r2_score"],
          "| Recall (VCI<35):", comparison["lgbm_day3_15_features"]["vci_t14"]["drought_recall_vci35"])
    print("4. LightGBM Day-3 15 Features (t+28 VCI) RMSE:", comparison["lgbm_day3_15_features"]["vci_t28"]["rmse_normalized"],
          "| R2:", comparison["lgbm_day3_15_features"]["vci_t28"]["r2_score"],
          "| SWDI t+14 RMSE:", comparison["lgbm_day3_15_features"]["swdi_t14_rmse"])

    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(models, os.path.join(models_dir, "lgbm_quantiles.joblib"))
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print("\n[Saved] Updated models/lgbm_quantiles.joblib and models/metrics.json")