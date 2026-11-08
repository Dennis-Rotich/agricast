import json
import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from split_and_baseline import (
    FEATURE_COLS, chronological_train_val_split,
    evaluate_predictions, generate_mock_telemetry
)

QUANTILES = [0.1, 0.5, 0.9]
TARGETS = ["target_vci_t14", "target_vci_t28", "target_swdi_t14"]

def train_rf_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, target_col: str = "target_vci_t14") -> dict:
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf_pipe.fit(train_df[FEATURE_COLS], train_df[target_col])
    preds = rf_pipe.predict(val_df[FEATURE_COLS])
    metrics = evaluate_predictions(val_df[target_col].values, preds, zonal_lookup_df=val_df)
    return {"model": rf_pipe, "metrics": metrics}

def train_lgbm_quantiles(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    models = {}
    val_preds = val_df[["grid_id", "lat", "lon", "date"]].copy()

    for target_col in TARGETS:
        models[target_col] = {}
        for alpha in QUANTILES:
            q_label = f"q{int(alpha * 100)}"
            lgbm = LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=350,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
            lgbm.fit(train_df[FEATURE_COLS], train_df[target_col])
            models[target_col][q_label] = lgbm
            val_preds[f"{target_col}_{q_label}"] = lgbm.predict(val_df[FEATURE_COLS])

    for target_col in TARGETS:
        q10 = val_preds[f"{target_col}_q10"].values
        q50 = val_preds[f"{target_col}_q50"].values
        q90 = val_preds[f"{target_col}_q90"].values
        val_preds[f"{target_col}_q10"] = np.minimum(q10, q50)
        val_preds[f"{target_col}_q90"] = np.maximum(q90, q50)
    
    val_preds["vulnerability_score_0_100"] = np.clip(
       100.0 - val_preds["target_vci_t14_q50"], 0.8, 100.0
    ).round(1)
    
    metrics_t14 = evaluate_predictions(
        y_true_vci=val_df["target_vci_t14"].values,
        y_pred_vci=val_preds["target_vci_t14_q50"].values,
        drought_trigger_pred=val_preds["target_vci_t14_q10"].values,
        zonal_lookup_df=val_preds
    )
    
    metrics_t28 = evaluate_predictions(
        y_true_vci=val_df["target_vci_t28"].values,
        y_pred_vci=val_preds["target_vci_t28_q50"].values,
        drought_trigger_pred=val_preds["target_vci_t28_q10"].values,
        zonal_lookup_df=val_preds
    )
    
    return models, {"t14": metrics_t14, "t28": metrics_t28}, val_preds

if __name__ == "__main__":
    data_path = "data/processed/pilot_features.parquet"
    if not os.path.exists(data_path):
        df = generate_mock_telemetry("data/sample/mock_features.parquet")
    else:
        df = pd.read_parquet(data_path)

    train_df, val_df = chronological_train_val_split(df, val_months=6, horizon_days=14)

    print("\n--- 1. Training Baseline Random Forest (t+14 VCI) ---")
    rf_res = train_rf_baseline(train_df, val_df, "target_vci_t14")
    print(json.dumps(rf_res["metrics"], indent=2))

    print("\n--- 2. Training Multi-Quantile LightGBM (alpha = 0.1, 0.5, 0.9) ---")
    lgbm_models, lgbm_metrics, val_preds = train_lgbm_quantiles(train_df, val_df)
    print("LightGBM t+14 Metrics:", json.dumps(lgbm_metrics["t14"], indent=2))
    print("LightGBM t+28 Metrics:", json.dumps(lgbm_metrics["t28"], indent=2))

    # Save Contract C artifacts for Member 3
    os.makedirs("models", exist_ok=True)
    joblib.dump(lgbm_models, "models/lgbm_quantiles.joblib")
    with open("models/metrics.json", "w") as f:
        json.dump({"rf_baseline_t14": rf_res["metrics"], "lgbm": lgbm_metrics}, f, indent=2)
    print("\n[Saved] models/lgbm_quantiles.joblib and models/metrics.json")