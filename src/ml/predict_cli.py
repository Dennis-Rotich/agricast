#!/usr/bin/env python3
"""AgriCast CLI: Query field-level crop vulnerability & drought stress forecasts.

Usage examples:
    # 1. Quick query by Grid ID (14-day forward default):
    python3 predict_cli.py --grid KAB_001

    # 2. Query by GPS coordinates with crop and growth stage:
    python3 predict_cli.py --lat -0.1736 --lon 35.9665 --crop maize --stage vegetative

    # 3. Query 28-day forward horizon:
    python3 predict_cli.py --grid KAB_042 --horizon 28 --crop sorghum --stage grain_filling
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, ROOT_DIR)

from train_telemetry import FEATURE_COLS, engineer_features, load_corridor_telemetry


def get_agronomic_directive(crop: str, stage: str, score: float) -> str:
    """Translates the 0-100 Crop Drought Vulnerability Score into growth-stage specific farming actions."""
    crop = crop.lower()
    stage = stage.lower()

    if "maize" in crop:
        if "veg" in stage:  # Vegetative (V4-V8)
            if score >= 60.0:
                return (
                    "[CRITICAL ACTION] Delay Urea/CAN top-dressing. Applying fertilizer under soil "
                    "moisture deficit causes severe root burning. Apply 5-7 cm crop residue mulch to preserve "
                    "sub-canopy humidity."
                )
            return "[NORMAL] Soil moisture reserve is adequate. Proceed with planned field cultivation."
        elif "tassel" in stage or "silk" in stage:  # Tasseling / Silking
            if score >= 70.0:
                return (
                    "[EMERGENCY TRIGGER] Critical yield-determining phase. Trigger priority supplemental "
                    "micro-irrigation directly to root zones. Weed immediately to eliminate transpiration competition."
                )
            return "[NORMAL] Moisture is currently sufficient for pollination. Monitor daily heat accumulation."

    elif "bean" in crop:  # Common Beans
        if score < 35.0:
            return (
                "[OPTIMAL CONDITIONS] Low Risk. Soil moisture reserve is sufficient. Proceed with planned weeding "
                "and foliar micronutrient feeding as scheduled."
            )
        return "[MODERATE RISK] Thin weed competition and apply light mulch to prevent premature flower abortion."

    elif "sorghum" in crop or "millet" in crop:  # Sorghum / Millet (Grain Filling)
        if score >= 50.0:
            return (
                "[ADAPTIVE ACTION] Moderate Deficit. Thin non-productive side-tillers to minimize whole-plant "
                "transpiration. Monitor field borders for fall armyworm migration triggered by elevated thermal stress."
            )
        return "[NORMAL] Crop moisture reserve is within seasonal tolerance thresholds."

    # General fallback rule
    if score >= 70.0:
        return "[HIGH DROUGHT STRESS] Suspend fertilizer inputs. Prioritize moisture conservation and emergency irrigation."
    elif score >= 40.0:
        return "[MODERATE STRESS] Apply sub-canopy mulching to reduce surface evaporation."
    return "[LOW STRESS] Optimal agronomic conditions. Proceed with standard seasonal calendar."


def run_cli():
    parser = argparse.ArgumentParser(
        description="AgriCast: Field-Scale Crop Vulnerability & Drought Stress Forecaster CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--grid", type=str, default=None, help="Grid ID in pilot corridor (e.g. KAB_001 to KAB_100)")
    parser.add_argument("--lat", type=float, default=None, help="Field latitude (e.g. -0.1736)")
    parser.add_argument("--lon", type=float, default=None, help="Field longitude (e.g. 35.9665)")
    parser.add_argument("--horizon", type=int, choices=[14, 28], default=14, help="Forecast horizon in days (default: 14)")
    parser.add_argument(
        "--crop",
        type=str,
        default="maize",
        choices=["maize", "beans", "sorghum", "millet"],
        help="Crop type (default: maize)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="vegetative",
        choices=["vegetative", "tasseling", "flowering", "grain_filling"],
        help="Crop phenological growth stage (default: vegetative)",
    )

    args = parser.parse_args()

    # 1. Load telemetry and models
    raw_df, _ = load_corridor_telemetry()
    clean_df = engineer_features(raw_df)

    models_path = os.path.join(ROOT_DIR, "models", "lgbm_quantiles.joblib")
    assert os.path.exists(models_path), "Missing models/lgbm_quantiles.joblib. Run tune_optuna.py first."
    models = joblib.load(models_path)

    # 2. Match field grid cell
    latest_all = clean_df.sort_values("date").groupby("grid_id", as_index=False).last()

    if args.grid:
        matched = latest_all[latest_all["grid_id"] == args.grid.upper()]
        if len(matched) == 0:
            print(f"Error: Grid '{args.grid}' not found. Available grids: KAB_001 to KAB_100.")
            sys.exit(1)
        row = matched.iloc[0]
    elif args.lat is not None and args.lon is not None:
        sq_dist = (latest_all["lat"] - args.lat) ** 2 + (latest_all["lon"] - args.lon) ** 2
        row = latest_all.iloc[sq_dist.argmin()]
    else:
        # Default to first grid cell if no arguments provided
        row = latest_all.iloc[0]

    # 3. Model Inference across quantiles
    horizon_str = f"t{args.horizon}"
    target_vci = f"target_vci_{horizon_str}"
    target_swdi = "target_swdi_t14"

    X = pd.DataFrame([row[FEATURE_COLS]])
    vci_q10 = float(models[target_vci]["q10"].predict(X)[0])
    vci_q50 = float(models[target_vci]["q50"].predict(X)[0])
    vci_q90 = float(models[target_vci]["q90"].predict(X)[0])
    swdi_q50 = float(models[target_swdi]["q50"].predict(X)[0])

    # Monotonicity
    vci_q10 = min(vci_q10, vci_q50)
    vci_q90 = max(vci_q90, vci_q50)

    # Standardized 0-100 Vulnerability Score
    vci_stress = 100.0 - vci_q50
    swdi_stress = np.clip((0.65 - swdi_q50) * 40.0, -15.0, 25.0)
    vuln_score = round(float(np.clip(0.85 * vci_stress + 0.15 * swdi_stress, 0.0, 100.0)), 1)

    # Risk Tier categorization
    if vuln_score >= 70.0:
        risk_tier = "CRITICAL / SEVERE RISK (Red)"
    elif vuln_score >= 35.0:
        risk_tier = "MODERATE RISK (Amber)"
    else:
        risk_tier = "LOW RISK / SAFE (Green)"

    # Actionable Directive
    directive = get_agronomic_directive(args.crop, args.stage, vuln_score)

    # 4. Display Formatted Terminal Output
    target_date = pd.to_datetime(row["date"]) + pd.Timedelta(days=args.horizon)

    print("\n" + "=" * 76)
    print(f"  AGRICAST FIELD EARLY WARNING: {row['grid_id']} ({row['lat']:.4f}°N, {row['lon']:.4f}°E)")
    print("=" * 76)
    print(f"  Current Observation Date : {pd.to_datetime(row['date']).strftime('%Y-%m-%d')}")
    print(f"  Forward Forecast Horizon : {args.horizon} Days (Target Date: {target_date.strftime('%Y-%m-%d')})")
    print(f"  Target Crop & Stage      : {args.crop.title()} ({args.stage.title()})")
    print("-" * 76)
    print("  PREDICTIVE DROUGHT STRESS METRICS:")
    print(f"  * Crop Drought Vulnerability Score : {vuln_score} / 100  -->  {risk_tier}")
    print(f"  * Deterministic Median VCI Forecast: {vci_q50:.1f} / 100 (Critical threshold: VCI < 35)")
    print(f"  * 80% Confidence Interval (q10-q90): [{vci_q10:.1f}  to  {vci_q90:.1f}]")
    print(f"  * Soil Water Deficit Index (SWDI)  : {swdi_q50:.2f} (Available Water: {row['awc_0_30cm']:.0f} mm/m)")
    print("-" * 76)
    print("  LOCAL BIOPHYSICAL SENSING TELEMETRY:")
    print(f"  * 30-Day Rainfall Deficit (SPEI-30): {row['spei_30']:+.2f} sigma (vs. 35-yr CHIRPS baseline)")
    print(f"  * Sentinel-2 Canopy Health (NDVI)  : {row['ndvi']:.3f} | Red-Edge Chlorophyll (NDRE): {row['ndre']:.3f}")
    print(f"  * 14-Day Red-Edge Decay Rate       : {row['delta_ndre_14d']:+.3f} (Pre-visual wilt velocity)")
    print(f"  * Vapor Pressure Deficit (VPD)     : {row['vpd']:.2f} kPa | Heat Stress (GDD): {row['gdd_cum']:.0f}°C-days")
    print("-" * 76)
    print("  ACTIONABLE AGRONOMIC DIRECTIVE:")
    print(f"  {directive}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    run_cli()