import sys, time
from pathlib import Path
import numpy as np, pandas as pd, folium, streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.advisory.rule_engine import STAGES, get_advisory, band

st.set_page_config(layout="wide", page_title="AgriCast")
st.markdown(
    "<style>footer {visibility: hidden;} .block-container {padding-top: 2rem;}</style>",
    unsafe_allow_html=True,
)
st.title("AgriCast")
st.caption(
    "Drought stress early warning for smallholder farms. Backtest demo on precomputed forecasts."
)


@st.cache_data
def load():
    return pd.read_parquet("data/sample/mock_forecasts.parquet")


@st.cache_data
def load_names():
    return pd.read_csv("data/sample/grid_names.csv").set_index("grid_id")["place_name"]


names = load_names()

df = load()

crop = st.sidebar.selectbox("Crop", list(STAGES))
stage = st.sidebar.selectbox("Growth stage", STAGES[crop])
horizon = st.sidebar.radio("Horizon (days)", [14, 28])
date = st.sidebar.selectbox(
    "Forecast date (backtest)", sorted(df.forecast_date.unique())
)

view = df[(df.horizon_days == horizon) & (df.forecast_date == date)].reset_index(
    drop=True
)

BAND_COLOR = {"low": "#2f8f5b", "moderate": "#d99a1e", "high": "#c0392b"}

left, right = st.columns([3, 2])
with left:
    m = folium.Map(
        location=[view.lat.mean(), view.lon.mean()], zoom_start=11, tiles="OpenStreetMap"
    )
    #folium.TileLayer(
    #    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    #    attr="Esri, Maxar, Earthstar Geographics",
    #).add_to(m)
    for r in view.itertuples():
        folium.CircleMarker(
            [r.lat, r.lon],
            radius=5,
            weight=0,
            fill=True,
            fill_color=BAND_COLOR[band(r.vulnerability_score_0_100)],
            fill_opacity=0.85,
        ).add_to(m)
    out = st_folium(m, height=520, returned_objects=["last_clicked"])

with right:
    click = out.get("last_clicked")
    if click:
        t0 = time.perf_counter()
        i = np.argmin((view.lat - click["lat"]) ** 2 + (view.lon - click["lng"]) ** 2)
        r = view.iloc[i]
        ms = (time.perf_counter() - t0) * 1000
        st.subheader(names.get(r.grid_id, r.grid_id))
        st.caption(f"Grid cell {r.grid_id}")
        c1, c2 = st.columns(2)
        c1.metric("Vulnerability", f"{r.vulnerability_score_0_100:.0f} / 100")
        c2.metric(
            "VCI forecast",
            f"{r.vci_q50:.0f}",
            f"{r.vci_q10:.0f} to {r.vci_q90:.0f}",
            delta_color="off",
        )
        st.write(f"Top driver: {r.shap_top1_feature}")
        adv = get_advisory(
            crop,
            stage,
            r.vulnerability_score_0_100,
            horizon,
            r.shap_top1_feature,
            r.grid_id,
        )
        st.write(f"Risk: {adv['band']}")
        st.write(adv["advice"])
        if adv["draft"]:
            st.caption("Draft advice, pending agronomist review.")
        st.code(adv["sms"], language=None)

        st.caption(f"Lookup time: {ms:.1f} ms. Backtest date, precomputed forecast.")
    else:
        st.write("Click the map to look up a cell.")
