import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import time

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from app.components import scorecard, sms_sim
from app.components.diagnostics import fan_chart, shap_chart
from src.advisory.glossary import GLOSSARY
from src.advisory.rule_engine import STAGES, band, get_advisory, risk_driver

st.set_page_config(layout="wide", page_title="AgriCast")
st.markdown(
    "<style>footer {visibility: hidden;} .block-container {padding-top: 2rem;}</style>",
    unsafe_allow_html=True,
)

BAND_COLOR = {"low": "#2f8f5b", "moderate": "#d99a1e", "high": "#c0392b"}


@st.cache_data
def load():
    """Real model output if present, otherwise the synthetic sample. Returns (df, filename, binned)."""
    real = ROOT / "data/predictions/zonal_forecasts.parquet"
    path = real if real.exists() else ROOT / "data/sample/mock_forecasts.parquet"
    d = pd.read_parquet(path)
    d["forecast_date"] = pd.to_datetime(d["forecast_date"])
    d["issue_date"] = pd.to_datetime(d["date"]) if "date" in d.columns else d["forecast_date"]
    if "horizon_days" not in d.columns:
        d["horizon_days"] = 14
    binned = False
    n_cells = d.grid_id.nunique()
    if (d.groupby("issue_date").grid_id.nunique() < n_cells).any():
        # cells are sampled on different days: group issue dates into weeks so every
        # cell appears in every week, keeping the first forecast per cell, horizon and week
        binned = True
        d["view_date"] = d.issue_date.dt.to_period("W-SUN").dt.start_time
        d = d.sort_values("issue_date").groupby(["grid_id", "horizon_days", "view_date"],
                                                as_index=False).head(1)
    else:
        d["view_date"] = d.issue_date
    return d.reset_index(drop=True), path.name, binned


@st.cache_data
def load_names():
    path = ROOT / "data/sample/grid_names.csv"
    if path.exists():
        return pd.read_csv(path).set_index("grid_id")["place_name"]
    return pd.Series(dtype=str)


df, source_file, binned = load()
names = load_names()
is_mock = "mock" in source_file

st.title("AgriCast")
if is_mock:
    st.caption("Drought stress early warning for smallholder farms. "
               "Demonstration data: forecasts are synthetic.")
else:
    st.caption("Drought stress early warning for smallholder farms. "
               "Backtest on the final six months of pilot data. Not a live forecast.")

# ---- sidebar ----
crop = st.sidebar.selectbox("Crop", list(STAGES))
stage = st.sidebar.selectbox("Growth stage", STAGES[crop])

horizons = sorted(int(h) for h in df.horizon_days.unique())
if len(horizons) > 1:
    horizon = st.sidebar.radio("Horizon (days)", horizons)
else:
    horizon = horizons[0]
    st.sidebar.markdown(f"**Horizon**  \n{horizon} days ahead")

dates = sorted(df.view_date.unique())
date_fmt = (lambda d: "Week of " + pd.Timestamp(d).strftime("%d %b %Y")) if binned \
    else (lambda d: pd.Timestamp(d).strftime("%d %b %Y"))
if len(dates) > 1:
    date = st.sidebar.selectbox("Forecast issued (backtest)", dates, format_func=date_fmt)
else:
    date = dates[0]
    st.sidebar.markdown(f"**Forecast issued (backtest)**  \n{date_fmt(date)}")

with st.sidebar.expander("What do these terms mean?"):
    for term, meaning in GLOSSARY.items():
        st.markdown(f"**{term}**  \n{meaning}")

view = df[(df.horizon_days == horizon) & (df.view_date == date)].reset_index(drop=True)

tab_map, tab_sim, tab_score = st.tabs(
    ["Map and forecast", "SMS and USSD simulator", "Model scorecard"])

# ---- map and forecast ----
with tab_map:
    left, right = st.columns([3, 2])

    with left:
        m = folium.Map(location=[view.lat.mean(), view.lon.mean()], zoom_start=12,
                       tiles="OpenStreetMap")
        for r in view.itertuples():
            folium.CircleMarker(
                [r.lat, r.lon], radius=5, weight=0, fill=True,
                fill_color=BAND_COLOR[band(r.vulnerability_score_0_100)],
                fill_opacity=0.85,
            ).add_to(m)
        legend = """
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
            background: #fafaf7; padding: 8px 12px; font-size: 13px;
            line-height: 1.6; color: #1f2a24;">
<b>Drought risk</b><br>
<span style="color:#2f8f5b;">&#9679;</span> Low (0 to 35)<br>
<span style="color:#d99a1e;">&#9679;</span> Moderate (36 to 69)<br>
<span style="color:#c0392b;">&#9679;</span> High (70 to 100)
</div>"""
        m.get_root().html.add_child(folium.Element(legend))
        out = st_folium(m, height=520, returned_objects=["last_clicked"], key="map")

    with right:
        click = out.get("last_clicked")
        if click:
            t0 = time.perf_counter()
            i = np.argmin((view.lat - click["lat"]) ** 2 + (view.lon - click["lng"]) ** 2)
            r = view.iloc[i]
            ms = (time.perf_counter() - t0) * 1000

            st.subheader(names.get(r.grid_id, r.grid_id))
            st.caption(f"Grid cell {r.grid_id}. Forecast issued {r.issue_date:%d %b %Y}, "
                       f"for {r.forecast_date:%d %b %Y}.")

            c1, c2 = st.columns(2)
            c1.metric("Drought risk score", f"{r.vulnerability_score_0_100:.0f} / 100")
            c2.metric("Crop health outlook (VCI)", f"{r.vci_q50:.0f}",
                      f"range {r.vci_q10:.0f} to {r.vci_q90:.0f}", delta_color="off")

            cell = df[(df.grid_id == r.grid_id) & (df.view_date == date)] \
                .sort_values("horizon_days")
            if len(cell) >= 2:
                st.plotly_chart(fan_chart(cell), key="fan")
            else:
                st.caption("The forecast range chart needs both the 14-day and 28-day "
                           "forecasts. This export has the 14-day forecast only.")

            adv = get_advisory(crop, stage, r.vulnerability_score_0_100, horizon,
                               risk_driver(r), r.grid_id)
            st.write(f"Risk: {adv['band']}")
            st.write(adv["advice"])
            if adv["draft"]:
                st.caption("Draft advice, pending agronomist review.")
            st.code(adv["sms"], language=None)

            st.plotly_chart(shap_chart(r), key="shap")
            st.caption("Red bars push risk up, green bars pull it down.")
            st.caption(f"Lookup time: {ms:.1f} ms. "
                       + ("Synthetic demo forecast." if is_mock else "Precomputed backtest forecast."))
        else:
            st.write("Click the map to look up a cell.")

# ---- SMS and USSD simulator ----
with tab_sim:
    sms_sim.render(view, names, crop, stage, horizon)

# ---- model scorecard ----
with tab_score:
    scorecard.render(ROOT)
