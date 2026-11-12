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

DRIVER_TEXT = {  # short phrases for SMS: describe a cause of stress
    "spei_30": "30d rain deficit",
    "spei_14": "14d rain deficit",
    "spei_30_lag7": "recent rain deficit",
    "spei30_x_awc": "rain and soil",
    "ndre": "crop stress signal",
    "ndre_lag7": "crop stress signal",
    "delta_ndre_14d": "crop health falling",
    "ndvi": "low greenness",
    "ndvi_lag7": "low greenness",
    "vpd": "dry air",
    "esi": "low soil moisture",
    "gdd_cum": "heat",
    "awc_0_30cm": "low soil water",
    "sand_clay_ratio": "sandy soil",
    "soc_0_30cm": "low soil carbon",
}

CHART_LABEL = {  # neutral names for the chart: the bar colour says if it raises or lowers risk
    "spei_30": "30-day rain balance",
    "spei_14": "14-day rain balance",
    "spei_30_lag7": "Rain balance last week",
    "spei30_x_awc": "Rain balance and soil",
    "ndre": "Crop stress signal",
    "ndre_lag7": "Crop stress signal last week",
    "delta_ndre_14d": "Change in crop stress signal",
    "ndvi": "Greenness",
    "ndvi_lag7": "Greenness last week",
    "vpd": "Dryness of air",
    "esi": "Soil moisture stress",
    "gdd_cum": "Heat build-up",
    "awc_0_30cm": "Soil water storage",
    "sand_clay_ratio": "Soil texture",
    "soc_0_30cm": "Soil carbon",
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
    msg = (
        f"AGRICAST [{plot_id}] {CROP_SHORT.get(crop, crop)} "
        f"{STAGE_SHORT.get(stage, stage)}: {horizon_days}d drought risk "
        f"{band(score).upper()} ({score:.0f}/100). {action}"
    )
    if driver and band(score) != "low":
        msg += f" Driver: {DRIVER_TEXT.get(driver, 'weather stress')}."
    return msg


def risk_driver(row):
    """First of the top 3 SHAP drivers that RAISES risk (positive value), else None."""
    for k in (1, 2, 3):
        if row[f"shap_top{k}_val"] > 0:
            return row[f"shap_top{k}_feature"]
    return None
