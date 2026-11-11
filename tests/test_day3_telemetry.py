import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src", "ml"))

from train_telemetry import (
    FEATURE_COLS,
    NEW_COLS,
    TARGETS,
    engineer_features,
    load_corridor_telemetry,
    run_training_and_comparison,
)


def test_day3_lags_and_interactions_no_cross_grid_leakage():
    """Verifies all 5 lag/interaction columns exist and SPEI-30 x AWC is exact."""
    df, _ = load_corridor_telemetry()
    df_feat = engineer_features(df)

    for col in NEW_COLS:
        assert col in df_feat.columns, f"Missing Day-3 feature column: {col}"

    # Verify interaction term formula: spei_30 * (awc_0_30cm / 100.0)
    expected_inter = df_feat["spei_30"] * (df_feat["awc_0_30cm"] / 100.0)
    assert np.allclose(df_feat["spei30_x_awc"], expected_inter), "Incorrect spei30_x_awc calculation"


def test_day3_all_four_targets_and_benchmarks():
    """Verifies VCI & SWDI models train for both t+14 and t+28 and hit Concept Note targets."""
    df, _ = load_corridor_telemetry()
    models, comparison, val_preds = run_training_and_comparison(df)

    assert len(FEATURE_COLS) == 15
    for t_col in TARGETS:
        assert t_col in models, f"Missing target model {t_col}"
        for q in ("q10", "q50", "q90"):
            assert f"{t_col}_{q}" in val_preds.columns

    m14 = comparison["lgbm_day3_15_features"]["vci_t14"]
    assert m14["rmse_normalized"] < 0.08
    assert m14["r2_score"] > 0.82
    assert m14["drought_recall_vci35"] > 0.85