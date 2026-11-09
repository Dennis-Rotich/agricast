import numpy as np, pandas as pd

rng = np.random.default_rng(42)
lats = np.linspace(0.95, 1.10, 10)   # placeholder box, swap for pilot_grid.geojson
lons = np.linspace(34.90, 35.10, 10)
dates = pd.to_datetime(["2026-03-01", "2026-03-15", "2026-03-29"])
feats = ["spei_30", "ndre", "vpd", "esi", "ndvi", "gdd_cum"]

rows = []
for i, (lat, lon) in enumerate((a, b) for a in lats for b in lons):
    for d in dates:
        for h in (14, 28):
            q50 = rng.uniform(15, 75)
            top = rng.choice(feats, 3, replace=False)
            vals = np.sort(rng.uniform(2, 20, 3))[::-1]
            rows.append({
                "grid_id": f"G{i:03d}", "lat": lat, "lon": lon,
                "forecast_date": d, "horizon_days": h,
                "vci_q10": max(q50 - rng.uniform(5, 12), 0),
                "vci_q50": q50,
                "vci_q90": min(q50 + rng.uniform(5, 12), 100),
                "swdi_q50": rng.uniform(-3, 3),
                "vulnerability_score_0_100": float(np.clip(100 - q50 + rng.normal(0, 5), 0, 100)),
                "drought_flag_vci35": q50 < 35,
                **{f"shap_top{k+1}_feature": top[k] for k in range(3)},
                **{f"shap_top{k+1}_val": vals[k] for k in range(3)},
            })
pd.DataFrame(rows).to_parquet("data/sample/mock_forecasts.parquet", index=False)
