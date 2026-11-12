import json
from pathlib import Path

import pandas as pd
import streamlit as st

TARGETS = {"rmse": 8.0, "r2": 0.82, "recall": 0.85, "ms": 400.0}  # concept note, RMSE in VCI points


def _find(root: Path):
    for p in (root / "models/metrics.json", root / "src/ml/models/metrics.json"):
        if p.exists():
            return p
    return None


def _row(label, d):
    rmse, r2 = d["rmse_0_100_scale"], d["r2_score"]
    rec, ms = d["drought_recall_vci35"], d["lookup_latency_ms"]
    met = rmse < TARGETS["rmse"] and r2 > TARGETS["r2"] and rec > TARGETS["recall"] and ms < TARGETS["ms"]
    return {"Model": label, "RMSE (VCI points)": round(rmse, 1), "R2": round(r2, 2),
            "Drought recall": f"{rec:.0%}", "Lookup (ms)": round(ms, 1),
            "Meets all targets": "Yes" if met else "No"}


def _holdout(root: Path):
    for p in (root / "models/holdout_metrics.json", root / "src/ml/models/holdout_metrics.json"):
        if p.exists():
            return json.load(open(p))
    return None


HOLDOUT_LABELS = [
    ("persistence", "Persistence baseline"),
    ("committed_model", "Model as served (settings tuned on a window that includes this one)"),
    ("retuned_q10_lt_35", "Model re-tuned on earlier months, alert below 35"),
    ("retuned_q10_fitted", "Model re-tuned, alert threshold fitted on earlier months"),
]


def render_holdout(h):
    st.subheader("Untouched hold-out check")
    rows = []
    for hz in ("14", "28"):
        for key, label in HOLDOUT_LABELS:
            r = h["results"][hz][key]
            name = f"{label} ({r['threshold']})" if key == "retuned_q10_fitted" else label
            rows.append({"Horizon": f"{hz}-day", "Model": name,
                         "RMSE (VCI points)": r["rmse_vci_points"], "R2": r["r2"],
                         "Drought recall": f"{r['drought_recall']:.0%}",
                         "Drought precision": f"{r['drought_precision']:.0%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    t0, t1 = h["test_window"]
    st.caption(f"Settings were tuned on earlier months only and tested on {t0} to {t1} "
               f"({h['test_rows']} cell-dates, about 14% of them below VCI 35). That is a small, "
               "spatially clustered sample, so treat these figures as indicative. Precision shows "
               "how many drought alerts were real: the rest are false alarms.")


def render(root: Path):
    path = _find(root)
    if path is None:
        st.write("No metrics.json found yet.")
        return
    m = json.load(open(path))
    day3 = m.get("lgbm_day3_15_features", {})
    rows = []
    for label, d in [
        ("Persistence baseline, 14-day", m.get("persistence_t14") or m.get("persistence_baseline_vci_t14")),
        ("Random forest, 14-day", m.get("rf_baseline_vci_t14")),
        ("LightGBM basic, 14-day", m.get("lgbm_base_10_features_vci_t14")),
        ("LightGBM full, 14-day", day3.get("vci_t14")),
        ("Persistence baseline, 28-day", m.get("persistence_t28")),
        ("LightGBM full, 28-day", day3.get("vci_t28")),
    ]:
        if d:
            rows.append(_row(label, d))

    st.subheader("Model scorecard")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Targets from the concept note: RMSE under 8 VCI points, R2 above 0.82, "
               "drought recall above 85%, lookup under 400 ms.")
    if not (m.get("persistence_t14") or m.get("persistence_baseline_vci_t14")):
        st.caption("Persistence baseline (predict that vegetation stays as it is now) is not in "
                   "metrics.json yet, so the gain over simply assuming no change is not shown.")
    else:
        st.caption("Persistence baseline = assume vegetation stays exactly as it is today. "
                   "A forecast is only useful if it beats this.")
    st.caption("Evaluated on the final six months of the pilot data, using an out-of-time split. "
               "Model settings were tuned on this same window, so read this table as optimistic. "
               "The hold-out check below is the cleaner test.")
    st.caption("Drought recall for the LightGBM rows uses the 10th percentile forecast as the "
               "trigger, so it catches more droughts and also raises more false alarms. "
               "Precision for this table is not reported; see the hold-out table below.")
    h = _holdout(root)
    if h:
        render_holdout(h)
    st.caption("VCI here is a smoothed, per-cell normalized NDVI index, not an official VCI product.")
