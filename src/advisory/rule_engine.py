"""Agronomic rule engine: crop + growth stage + vulnerability score -> advice.

All advice text is draft until an agronomist has reviewed it.
"""

STAGES = {
    "Maize": ["Vegetative (V4-V8)", "Tasseling / Silking"],
    "Common Beans": ["Pre-Flowering"],
    "Sorghum / Millet": ["Grain Filling"],
}

CROP_SHORT = {"Maize": "Maize", "Common Beans": "Beans", "Sorghum / Millet": "Sorghum"}
STAGE_SHORT = {
    "Vegetative (V4-V8)": "V4-V8",
    "Tasseling / Silking": "Silking",
    "Pre-Flowering": "Pre-flower",
    "Grain Filling": "Grain fill",
}
DRIVER_TEXT = {
    "spei_30": "30d rain deficit",
    "spei_14": "14d rain deficit",
    "ndre": "crop stress signal",
    "ndvi": "low greenness",
    "vpd": "dry air",
    "esi": "low soil moisture",
    "gdd_cum": "heat",
}

# (crop, stage, band) -> (full advice, short SMS action)
CARDS = {
    ("Maize", "Vegetative (V4-V8)", "moderate"): (
        "Delay Urea/CAN top-dressing. Fertilizer application under soil moisture "
        "deficit causes root burning. Apply 5-7 cm crop residue mulch to preserve "
        "sub-canopy humidity.",
        "Do NOT apply Urea/CAN. Mulch 5-7cm.",
    ),
    ("Maize", "Tasseling / Silking", "high"): (
        "Priority micro-irrigation trigger. Critical yield-determining phase. Ration "
        "supplemental irrigation directly to root zones; weed immediately to "
        "eliminate transpiration competition.",
        "Irrigate root zone now. Weed today.",
    ),
    ("Common Beans", "Pre-Flowering", "low"): (
        "Optimal operational conditions. Soil moisture reserve is sufficient. Proceed "
        "with planned weeding and foliar micronutrient feeding as scheduled.",
        "Conditions good. Weed and feed as planned.",
    ),
    ("Sorghum / Millet", "Grain Filling", "moderate"): (
        "Thin non-productive side-tillers to minimize whole-plant transpiration. "
        "Monitor field borders for fall armyworm migration triggered by elevated "
        "thermal stress.",
        "Thin side-tillers. Watch borders for armyworm.",
    ),
}

# Fallback when no crop-specific card exists for the combination.
GENERIC = {
    "low": (
        "Conditions favorable. Continue planned field operations.",
        "Conditions good. Continue as planned.",
    ),
    "moderate": (
        "Moisture stress is building. Delay fertilizer application and mulch the plot "
        "to hold soil moisture.",
        "Stress building. Delay fertilizer. Mulch.",
    ),
    "high": (
        "Severe stress likely. Prioritize irrigation or moisture conservation on this "
        "plot and contact your extension officer.",
        "Severe stress. Irrigate or mulch. Call extension officer.",
    ),
}


def band(score: float) -> str:
    if not 0 <= score <= 100:
        raise ValueError(f"score must be 0-100, got {score}")
    if score < 36:
        return "low"
    if score < 70:
        return "moderate"
    return "high"


def get_advisory(crop, stage, score, horizon_days=14, driver=None, plot_id="PLOT"):
    b = band(score)
    key = (crop, stage, b)
    source = "card" if key in CARDS else "generic"
    advice, sms_action = CARDS.get(key, GENERIC[b])
    return {
        "band": b,
        "advice": advice,
        "sms": build_sms(plot_id, crop, stage, score, horizon_days, driver, sms_action),
        "source": source,
        "draft": True,  # flip only after agronomist review
    }


def build_sms(plot_id, crop, stage, score, horizon_days, driver, action):
    drv = DRIVER_TEXT.get(driver, "weather stress")
    return (
        f"AGRICAST [{plot_id}] {CROP_SHORT.get(crop, crop)} "
        f"{STAGE_SHORT.get(stage, stage)}: {horizon_days}d drought risk "
        f"{band(score).upper()} ({score:.0f}/100). {action} Driver: {drv}."
    )
