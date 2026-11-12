import json
import os
import sys
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import precision_score, recall_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_LGBM = False

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR,"..","..")) if THIS_DIR.endswith(os.path.join("src","ml")) else THIS_DIR
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, ROOT_DIR)

from split_and_baseline import chronological_train_val_split, evaluate_predictions
from train_telemetry import (
    FEATURE_COLS,
    TARGETS,
    QUANTILES,
    engineer_features,
    load_corridor_telemetry,
)

def build_tuned_estimator(params: dict, alpha: float):
    if HAS_LGBM:
        return LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=params.get("n_estimators", 350),
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=42,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=alpha,
        max_iter=params.get("n_estimators", 350),
        learning_rate=params["learning_rate"],
        max_leaf_nodes=params["num_leaves"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_child_samples"],
        random_state=42,
    )

def calibrate_drought_threshold(
    y_true_vci: np.ndarray,
    pred_vci_q10: np.ndarray,
    min_recall: float = 0.85,
) -> dict:
    y_true_binary = (y_true_vci < 35.0).astype(int)
    thresholds = np.linspace(30.0, 45.0, 61)
    best_candidate = None

    for thresh in thresholds:
        pred_binary = (pred_vci_q10 < thresh).astype(int)
        rec = float(recall_score(y_true_binary, pred_binary, zero_division=0))
        prec = float(precision_score(y_true_binary, pred_binary, zero_division=0))

        if rec >= min_recall:
            f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
            if best_candidate is None or f1 > best_candidate["f1_score"]:
                best_candidate = {
                    "calibrated_trigger_threshold": round(float(thresh), 2),
                    "drought_recall": round(rec, 4),
                    "drought_precision": round(prec, 4),
                    "f1_score": round(f1, 4),
                }
        
        if best_candidate is None:
            best_candidate = {
                "calibrated_trigger_threshold": 35.0,
                "drought_recall": float(recall_score(y_true_binary, (pred_vci_q10 < 35.0).astype(int))),
                "drought_predcision": float(precision_score(y_true_binary, (pred_vci_q10 < 35.0).astype(int))),
                "f1_score": 0.0,
            }
        return best_candidate

def run_optuna_study(train_df: pd.DataFrame, val_df: pd.DataFrame, n_trials: int = 50) -> dict:
    X_train, y_train_t14 = train_df[FEATURE_COLS], train_df["target_vci_t14"]
    y_train_t28 = train_df["target_vci_t28"]
    X_val, y_val_t14 = val_df[FEATURE_COLS], val_df["target_vci_t14"]
    y_val_t28 = val_df["target_vci_t28"]

    def objective(trial: optuna.Trial) -> float:
        params = {
           "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "n_estimators": 350, 
        }

        m_14 = build_tuned_estimator(params, 0.5).fit(X_train, y_train_t14)
        m_28 = build_tuned_estimator(params, 0.5).fit(X_train, y_train_t28)

        p14 = m_14.predict(X_val)
        p28 = m_28.predict(X_val)

        rmse14 = np.sqrt(np.mean(((y_val_t14 - p14) / 100.0) ** 2))
        rmse28 = np.sqrt(np.mean(((y_val_t28 - p28) / 100.0) ** 2))

        return float(0.6 * rmse14 + 0.4 * rmse28)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    print(f"[Optuna] Launching {n_trials} trials across 15 features...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[Optuna] Best trial score: {study.best_value:.4f}")
    print("[Optuna] Best hyperparameters:", json.dumps(study.best_params, indent=2))
    return study.best_params

def train_production_bundle(df: pd.DataFrame, best_params: dict) -> tuple[dict, dict, pd.DataFrame]:
    df_feat = engineer_features(df)
    train_df, val_df = chronological_train_val_split(df_feat, val_months=6, horizon_days=14)

    models = {}
    val_preds = val_df[["grid_id", "lat", "lon", "date"]].copy()

    for target_col in TARGETS:
        models[target_col] = {}
        for alpha in QUANTILES:
            q_label = f"q{int(alpha * 100)}"
            est = build_tuned_estimator(best_params, alpha)
            est.fit(train_df[FEATURE_COLS], train_df[target_col])
            models[target_col][q_label] = est
            val_preds[f"{target_col}_{q_label}"] = est.predict(val_df[FEATURE_COLS]) 

        q10 = val_preds[f"{target_col}_q10"].values
        q50 = val_preds[f"{target_col}_q50"].values
        q90 = val_preds[f"{target_col}_q90"].values
        val_preds[f"{target_col}_q10"] = np.minimum(q10, q50)
        val_preds[f"{target_col}_q90"] = np.maximum(q90, q50)

    calib_t14 = calibrate_drought_threshold(val_df["target_vci_t14"].values, val_preds["target_vci_t14_q10"].values, min_recall=0.85)
    calib_t28 = calibrate_drought_threshold(val_df["target_vci_t28"].values, val_preds["target_vci_t28_q10"].values, min_recall=0.85)

    vci_stress = 100.0 - val_preds["target_vci_t14_q50"]
    swdi_stress = np.clip((0.65 - val_preds["target_swdi_t14_q50"]) * 40.0, -15.0, 25.0)
    val_preds["vulnerability_score_0_100"] = np.clip(0.85 * vci_stress + 0.15 * swdi_stress, 0.0, 100.0).round(1)

    thresh_14 = calib_t14["calibrated_trigger_threshold"]
    val_preds["drought_flag_vci35"] = (val_preds["target_vci_t14_q10"] < thresh_14).astype(int)

    metrics = {
        "best_hyperparameters": best_params,
        "vci_t14_eval": evaluate_predictions(
            val_df["target_vci_t14"].values,
            val_preds["target_vci_t14_q50"].values,
            drought_trigger_pred=val_preds["target_vci_t14_q10"].values,
            zonal_lookup_df=val_preds,
        ),
        "vci_t28_eval": evaluate_predictions(
            val_df["target_vci_t28"].values,
            val_preds["target_vci_t28_q50"].values,
            drought_trigger_pred=val_preds["target_vci_t28_q10"].values,
            zonal_lookup_df=val_preds,
        ),
        "calibration_t14": calib_t14,
        "calibration_t28": calib_t28,
        "swdi_t14_rmse": round(float(np.sqrt(np.mean((val_df["target_swdi_t14"] - val_preds["target_swdi_t14_q50"]) ** 2))), 4),
        "swdi_t28_rmse": round(float(np.sqrt(np.mean((val_df["target_swdi_t28"] - val_preds["target_swdi_t28_q50"]) ** 2))), 4),
    }

    return models, metrics, val_preds

if __name__ == "__main__":
    raw_df, src_name = load_corridor_telemetry()
    print(f"[Day 4] Loaded {len(raw_df):,} rows from {src_name}")

    clean_df = engineer_features(raw_df)
    train_data, val_data = chronological_train_val_split(clean_df, val_months=6, horizon_days=14)

    best_hp = run_optuna_study(train_data, val_data, n_trials=50)

    print("\n--- Training Final Tuned Production Models ---")
    prod_models, prod_metrics, val_predictions = train_production_bundle(raw_df, best_hp)

    m14 = prod_metrics["vci_t14_eval"]
    m28 = prod_metrics["vci_t28_eval"]
    c14 = prod_metrics["calibration_t14"]

    print("\n================ DAY 4 FINAL SCORECARD ================")
    print(f"1. 14-Day VCI RMSE (target < 0.08) : {m14['rmse_normalized']:.4f} [{'PASS' if m14['rmse_normalized'] < 0.08 else 'FAIL'}]")
    print(f"2. 14-Day VCI R²   (target > 0.82) : {m14['r2_score']:.4f} [{'PASS' if m14['r2_score'] > 0.82 else 'FAIL'}]")
    print(f"3. 14-Day Drought Recall (target > 85%): {c14['drought_recall'] * 100:.2f}% [{'PASS' if c14['drought_recall'] >= 0.85 else 'FAIL'}]")
    print(f"   -> Calibrated trigger threshold  : q10 < {c14['calibrated_trigger_threshold']} (Precision: {c14['drought_precision'] * 100:.2f}%)")
    print(f"4. 28-Day VCI RMSE                 : {m28['rmse_normalized']:.4f} | R²: {m28['r2_score']:.4f}")
    print(f"5. Single-Coordinate Lookup Latency : {m14['lookup_latency_ms']:.2f} ms (< 400ms target)")
    print("=======================================================")

    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(prod_models, os.path.join(models_dir, "lgbm_quantiles.joblib"))
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(prod_metrics, f, indent=2)

    print("\n[Saved] Updated models/lgbm_quantiles.joblib and models/metrics.json")