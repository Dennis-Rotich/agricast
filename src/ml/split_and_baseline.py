import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, recall_score

FEATURE_COLS = [
    "ndvi", "ndre", "spei_14", "spei_30",
    "gdd_cum", "vpd", "esi",
    "awc_0_30cm", "sand_clay_ratio", "soc_0_30cm",
]

def generate_mock_telemetry(out_path: str = "data/sample/mock_features.parquet") -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="5D")
    n_grids = 40
    records = []

    for grid_id in range(1, n_grids + 1):
        lat = -0.18 + (grid_id // 8) * 0.01
        lon = 35.85 + (grid_id % 8) * 0.01
        awc = np.random.uniform(90, 180)
        sand_clay = np.random.uniform(0.8, 2.5)
        soc = np.random.uniform(1.1, 2.8)

        t = np.arange(len(dates))
        seasonal = np.sin(2 * np.pi * t / 36.5)
        spei_30 = seasonal + np.random.normal(0, 0.45, len(dates))
        spei_14 = 0.85 * spei_30 + np.random.normal(0, 0.3, len(dates))
        vpd = np.clip(1.6 - 0.55 * spei_14 + np.random.normal(0, 0.2, len(dates)), 0.4, 3.5)
        esi = np.clip(0.55 + 0.22 * spei_14 - 0.1 * vpd + np.random.normal(0, 0.03, len(dates)), 0.12, 0.78)
        gdd_cum = np.clip(120 + 45 * seasonal + np.random.normal(0, 15, len(dates)), 0.18, 0.90)

        ndre = np.clip(0.42 + 0.14 * spei_30 - 0.06 * vpd + np.random.normal(0, 0.03, len(dates)), 0.12, 0.78)
        ndvi = np.clip(ndre + 0.16 + np.random.normal(0, 0.03, len(dates)), 0.18, 0.90)

        base_stress_signal = (
            50.0 + 18.0 * spei_30 + 6.0 * spei_14 + 42.0 * (ndre - 0.40) - 9.0 * (vpd - 1.5) + 12.0 * (esi - 0.55) + 0.09 * (awc - 130)
        )

        vci_t14 = np.clip(base_stress_signal + np.random.normal(0, 3.2, len(dates)), 5.0, 98.0)
        vci_t28 = np.clip(0.92 * base_stress_signal + 4.0 + np.random.normal(0, 4.0, len(dates)), 5.0, 98.0)
        swdi_t14 = np.clip((vci_t14 - 50.0) / 15.0 + np.random.normal(0, 0.15, len(dates)), -3.5, 2.5)

        cloud_mask = np.random.rand(len(dates)) < 0.18
        ndvi_obs = np.where(cloud_mask, np.nan, ndvi)
        ndre_obs = np.where(cloud_mask, np.nan, ndre)

        df_grid = pd.DataFrame({
            "grid_id": f"GRID_{grid_id:03d}",
            "lat": lat,
            "lon": lon,
            "date": dates,
            "ndvi": ndvi_obs,
            "ndre": ndre_obs,
            "spei_14": spei_14,
            "spei_30": spei_30,
            "gdd_cum": gdd_cum,
            "vpd": vpd,
            "esi": esi,
            "awc_0_30cm": awc,
            "sand_clay_ratio": sand_clay,
            "soc_0_30cm": soc,
            "target_vci_t14": vci_t14,
            "target_vci_t28": vci_t28,
            "target_swdi_t14": swdi_t14,
        })

        records.append(df_grid.iloc[:-6])

    full_df = pd.concat(records, ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    full_df.to_parquet(out_path, index=False)
    print(f"[Mock Data] Saved {len(full_df):,} rows to {out_path}")
    return full_df

def chronological_train_val_split(df: pd.DataFrame, val_months: int = 6, horizon_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "grid_id"]).reset_index(drop=True)

    max_date = df["date"].max()
    val_start_date = max_date - pd.DateOffset(months=val_months)

    train_end_date = val_start_date - pd.Timedelta(days=horizon_days)

    train_df = df[df["date"] <= train_end_date].copy()
    val_df = df[df["date"] >= val_start_date].copy()

    assert train_df["date"].max() < val_df["date"].min(), "Data leakage detected!"
    print(
        f"[Out-of-Time Split] Train: {train_df['date'].min().date()} -> {train_df['date'].max().date()} "
        f"({len(train_df):,} rows) | Val: {val_df['date'].min().date()} -> {val_df['date'].max().date()} "
        f"({len(val_df):,} rows)"
    )
    return train_df, val_df

def evaluate_predictions(y_true_vci: np.ndarray, y_pred_vci:np.ndarray, drought_trigger_pred: np.ndarray = None, zonal_lookup_df: pd.DataFrame = None) -> dict:
    rmse_norm = float(np.sqrt(mean_squared_error(y_true_vci / 100.0, y_pred_vci / 100.0)))
    rmse_100 = float(np.sqrt(mean_squared_error(y_true_vci, y_pred_vci)))
    r2 = float(r2_score(y_true_vci, y_pred_vci))

    y_true_drought = (y_true_vci < 35.0).astype(int)
    if drought_trigger_pred is None:
        y_pred_drought = (y_pred_vci < 35.0).astype(int)
    else:
        y_pred_drought = (drought_trigger_pred < 35.0).astype(int)

    recall_vci35 = float(recall_score(y_true_drought, y_pred_drought, zero_division=0))

    latency_ms = 0.0
    if zonal_lookup_df is not None and len(zonal_lookup_df) > 0:
        sample_lat = zonal_lookup_df["lat"].iloc[0]
        sample_lon = zonal_lookup_df["lon"].iloc[0]
        t0 = time.perf_counter()
        _ = zonal_lookup_df.loc[
            ((zonal_lookup_df["lat"] - sample_lat) ** 2 + (zonal_lookup_df["lon"] - sample_lon) ** 2).idxmin()
        ]
        latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "rmse_normalized": round(rmse_norm, 4),
        "rmse_0_100_scale": round(rmse_100, 2),
        "r2_score": round(r2, 4),
        "drought_recall_vci35": round(recall_vci35, 4),
        "lookup_latency_ms": round(latency_ms, 3),
        "targets_met": {
            "rmse_below_0_08": rmse_norm < 0.08,
            "r2_above_0_82": r2 > 0.82,
            "recall_above_0_85": recall_vci35 > 0.85,
            "latency_below_400ms": latency_ms < 400.0,
        },
    }

if __name__ == "__main__":
    data_path = "data/processed/pilot_features.parquet"
    if not os.path.exists(data_path):
        df = generate_mock_telemetry("data/sample/mock_features.parquet")
    else:
        df = pd.read_parquet(data_path)

    train_df, val_df = chronological_train_val_split(df, val_months=6, horizon_days=14)