"""Check a forecasts parquet against Cotract B and the questions for Dennis.
Usage: python contracts/check_forecasts.py [path/to/forecasts.parquet] [path/to/metrics.json]
"""
import json
import sys
from pathlib import Path

import pandas as pd

B_COLS = ["grid_id", "lat", "lon", "forecast_date", "horizon_days", "vci_q10", "vci_q50",
          "vci_q90", "swdi_q50", "vulnerability_score_0_100", "drought_flag_vci35",
          "shap_top1_feature", "shap_top1_val", "shap_top2_feature", "shap_top2_val",
          "shap_top3_feature", "shap_top3_val"]
C_KEYS = ["rmse", "r2", "recall_vci35", "inference_ms"]


def say(n, ok, msg):
    print(f"[{'OK ' if ok else 'FIX'}] Q{n}: {msg}")


path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/predictions/zonal_forecasts.parquet")
df = pd.read_parquet(path)
print(f"{path}: {df.shape[0]} rows\n")

missing = [c for c in B_COLS if c not in df.columns]
say(0, not missing, f"missing columns: {missing}" if missing else "all Contract B columns present")
if missing:
    sys.exit("Fix the columns first; the other checks need them.")

bad = ((df.vci_q10 > df.vci_q50) | (df.vci_q50 > df.vci_q90)).sum()
say(2, bad == 0, f"{bad} rows with crossing quantiles")

a1, a2, a3 = df.shap_top1_val.abs(), df.shap_top2_val.abs(), df.shap_top3_val.abs()
bad = ((a1 < a2) | (a2 < a3)).sum()
say(3, bad == 0, f"{bad} rows where SHAP drivers are not sorted by absolute size")

s = df.vulnerability_score_0_100
say(5, s.between(0, 100).all(), f"score range {s.min():.1f} to {s.max():.1f}; "
    f"bands low/moderate/high = {(s < 36).sum()}/{((s >= 36) & (s < 70)).sum()}/{(s >= 70).sum()}")

print(f"      Q6 info: forecast dates {df.forecast_date.min().date()} to "
      f"{df.forecast_date.max().date()} ({df.forecast_date.nunique()} dates), "
      f"horizons {[int(h) for h in sorted(df.horizon_days.unique())]}. "
      "Confirm these sit in the held-out period.")

ids = df.grid_id.astype(str)
say(7, ids.str.match(r"^KAB_\d{3}$").all(),
    f"{ids.nunique()} grid ids, examples {list(ids.unique()[:3])}; Morgan's files use KAB_001 style")
geo = Path("data/processed/corridor_layers.geojson")
if geo.exists():
    g = {f["properties"]["grid_id"] for f in json.load(open(geo))["features"]}
    extra, absent = set(ids) - g, g - set(ids)
    say(7, not extra and not absent, f"ids not in geojson: {len(extra)}; geojson ids not forecast: {len(absent)}")

expected = df.grid_id.nunique() * df.forecast_date.nunique() * df.horizon_days.nunique()
dups = df.duplicated(["grid_id", "forecast_date", "horizon_days"]).sum()
say(8, len(df) == expected and dups == 0, f"{len(df)} rows vs {expected} expected, {dups} duplicates")

v = pd.concat([df.shap_top1_val, df.shap_top2_val, df.shap_top3_val])
print(f"      Q1 info: SHAP values range {v.min():.2f} to {v.max():.2f}, {(v < 0).mean():.0%} negative. "
      "Units and sign still need Dennis's answer.")

mpath = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("models/metrics.json")
if mpath.exists():
    m = json.load(open(mpath))
    miss = [k for k in C_KEYS if k not in m]
    say(11, not miss, f"metrics.json keys missing: {miss}" if miss else f"keys present: {sorted(m)}")
    print(f"      Q11 info: baselines in file: {[k for k in m if 'persist' in k or 'climat' in k] or 'none'}")
else:
    say(11, False, f"{mpath} not found")
