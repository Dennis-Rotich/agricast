import itertools
import pytest
from src.advisory.rule_engine import CARDS, DRIVER_TEXT, STAGES, band, get_advisory, risk_driver


def test_band_edges():
    assert band(0) == "low" and band(35.9) == "low"
    assert band(36) == "moderate" and band(69.9) == "moderate"
    assert band(70) == "high" and band(100) == "high"


@pytest.mark.parametrize("bad", [-1, 100.1])
def test_band_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        band(bad)


@pytest.mark.parametrize("crop,stage,score", [
    ("Maize", "Vegetative (V4-V8)", 62),
    ("Maize", "Tasseling / Silking", 84),
    ("Common Beans", "Pre-Flowering", 28),
    ("Sorghum / Millet", "Grain Filling", 55),
])
def test_concept_note_cards(crop, stage, score):
    out = get_advisory(crop, stage, score)
    assert out["source"] == "card"
    assert out["advice"] == CARDS[(crop, stage, out["band"])][0]


def test_fallback_is_generic():
    assert get_advisory("Maize", "Vegetative (V4-V8)", 90)["source"] == "generic"


def test_sms_never_exceeds_160_chars():
    drivers = list(DRIVER_TEXT) + ["unknown_feature", None]
    for crop, stages in STAGES.items():
        for stage, score, drv in itertools.product(stages, [10, 50, 95], drivers):
            sms = get_advisory(crop, stage, score, 28, drv, plot_id="PLOT-0042")["sms"]
            assert len(sms) <= 160, (len(sms), sms)


def test_everything_is_marked_draft():
    assert get_advisory("Maize", "Tasseling / Silking", 84)["draft"] is True


def test_sms_omits_driver_when_risk_is_low_or_driver_missing():
    assert "Driver" not in get_advisory("Common Beans", "Pre-Flowering", 20, 14, "ndvi")["sms"]
    assert "Driver" not in get_advisory("Maize", "Tasseling / Silking", 84, 14, None)["sms"]
    assert "Driver: 30d rain deficit." in get_advisory("Maize", "Tasseling / Silking", 84, 14, "spei_30")["sms"]


def test_risk_driver_skips_features_that_lower_risk():
    row = {"shap_top1_feature": "ndvi", "shap_top1_val": -23.4,
           "shap_top2_feature": "delta_ndre_14d", "shap_top2_val": 7.7,
           "shap_top3_feature": "ndre", "shap_top3_val": -4.0}
    assert risk_driver(row) == "delta_ndre_14d"
    row["shap_top2_val"] = -1.0
    assert risk_driver(row) is None
