"""Honest hold-out check. Does NOT touch models/ or metrics.json.

Tunes only on April to early August 2025, then reports on September to October 2025,
a window the tuning never saw. Run from the repo root:  python src/ml/holdout_eval.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, r2_score, recall_score

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, THIS_DIR)

from split_and_baseline import chronological_train_val_split
from train_telemetry import FEATURE_COLS, engineer_features, load_corridor_telemetry
from tune_optuna import build_tuned_estimator, run_optuna_study

TEST_START = pd.Timestamp("2025-09-01")
EMBARGO_DAYS = 28            # tune rows' targets must not reach into the test window


def metrics(y, p, trigger=None, thr=35.0):
    out = {"rmse_vci_points": round(float(np.sqrt(np.mean((y - p) ** 2))), 2),
           "r2": round(float(r2_score(y, p)), 3)}
    t = p if trigger is None else trigger
    yb, pb = (y < 35).astype(int), (t < thr).astype(int)
    out["drought_recall"] = round(float(recall_score(yb, pb, zero_division=0)), 3)
    out["drought_precision"] = round(float(precision_score(yb, pb, zero_division=0)), 3)
    return out


def calibrate(y, q10, min_recall=0.85):
    """Fixed version: scan the whole grid, keep the best F1 that still reaches the recall target."""
    yb, best = (y < 35).astype(int), (35.0, -1.0)
    for thr in np.linspace(30, 45, 61):
        pb = (q10 < thr).astype(int)
        r, p = recall_score(yb, pb, zero_division=0), precision_score(yb, pb, zero_division=0)
        f1 = 2 * p * r / (p + r + 1e-8)
        if r >= min_recall and f1 > best[1]:
            best = (float(thr), f1)
    return best[0]


df, _ = load_corridor_telemetry()
feat = engineer_features(df)
train, val = chronological_train_val_split(feat, val_months=6, horizon_days=14)
val["date"] = pd.to_datetime(val["date"])
tune = val[val.date <= TEST_START - pd.Timedelta(days=EMBARGO_DAYS)]
test = val[val.date >= TEST_START]
print(f"train {len(train):,} | tune {len(tune):,} ({tune.date.min().date()} to {tune.date.max().date()}) "
      f"| test {len(test):,} ({test.date.min().date()} to {test.date.max().date()})")

params = run_optuna_study(train, tune, n_trials=int(os.environ.get("TRIALS", 50)))

results = {}
for h in (14, 28):
    tgt = f"target_vci_t{h}"
    m10 = build_tuned_estimator(params, 0.1).fit(train[FEATURE_COLS], train[tgt])
    m50 = build_tuned_estimator(params, 0.5).fit(train[FEATURE_COLS], train[tgt])
    thr = calibrate(tune[tgt].values, m10.predict(tune[FEATURE_COLS]))
    y = test[tgt].values
    p50, p10 = m50.predict(test[FEATURE_COLS]), m10.predict(test[FEATURE_COLS])
    persist = test["ndvi"].values * 100.0
    # the model Dennis already committed (tuned on the whole six-month window, test included)
    old = joblib.load(os.path.join(ROOT_DIR, "models", "lgbm_quantiles.joblib"))[tgt]
    o50, o10 = old["q50"].predict(test[FEATURE_COLS]), old["q10"].predict(test[FEATURE_COLS])
    results[h] = {
        "persistence": metrics(y, persist),
        "committed_model": metrics(y, o50, o10),          # tuned on a window that includes this one
        "retuned_q10_lt_35": metrics(y, p50, p10, 35.0),
        "retuned_q10_fitted": {**metrics(y, p50, p10, thr), "threshold": round(thr, 1)},
        "share_below_35": round(float((y < 35).mean()), 3),
    }

import json
summary = {
    "tune_window": [str(tune.date.min().date()), str(tune.date.max().date())],
    "test_window": [str(test.date.min().date()), str(test.date.max().date())],
    "test_rows": int(len(test)),
    "best_params": params,
    "results": {str(h): r for h, r in results.items()},
}
with open(os.path.join(ROOT_DIR, "models", "holdout_metrics.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nWrote models/holdout_metrics.json (metrics.json and the model file are untouched)")
for h, r in results.items():
    print(f"\n=== {h}-day, test window {test.date.min().date()} to {test.date.max().date()} ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
