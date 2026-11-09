import itertools
import pytest
from src.advisory.rule_engine import CARDS, STAGES, band, get_advisory


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
    drivers = ["spei_30", "ndre", "unknown_feature", None]
    for crop, stages in STAGES.items():
        for stage, score, drv in itertools.product(stages, [10, 50, 95], drivers):
            sms = get_advisory(crop, stage, score, 28, drv, plot_id="PLOT-0042")["sms"]
            assert len(sms) <= 160, (len(sms), sms)


def test_everything_is_marked_draft():
    assert get_advisory("Maize", "Tasseling / Silking", 84)["draft"] is True
