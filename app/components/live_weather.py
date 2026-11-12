import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M"
DAYS_BACK = 20
SHOW_DAYS = 14


@st.cache_data(ttl=3600, show_spinner=False)
def fetch(lat: float, lon: float):
    """Recent daily weather from NASA POWER. Failures raise, so they are never cached."""
    end = dt.date.today()
    start = end - dt.timedelta(days=DAYS_BACK)
    resp = requests.get(
        API,
        params={"parameters": PARAMS, "community": "AG", "latitude": lat, "longitude": lon,
                "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
                "format": "JSON"},
        timeout=15,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.where(df > -900).dropna(how="all").sort_index()   # NASA marks missing days as -999
    return df, pd.Timestamp.now().strftime("%d %b %Y %H:%M")


def render(cell):
    """cell: dict with name, lat, lon of the last clicked map cell, or None."""
    if not cell:
        st.write("Click a cell on the map first, then open this tab.")
        return
    st.subheader(f"Live weather: {cell['name']}")
    try:
        df, fetched = fetch(round(cell["lat"], 2), round(cell["lon"], 2))
    except Exception:
        st.write("Live weather could not be loaded (no connection to NASA POWER). "
                 "The other tabs are not affected.")
        return
    df = df.tail(SHOW_DAYS)
    if df.empty or "PRECTOTCORR" not in df:
        st.write("NASA POWER has no data for this period yet.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Rain, last {len(df)} days", f"{df['PRECTOTCORR'].sum():.0f} mm")
    if "T2M_MAX" in df:
        c2.metric("Average daily high", f"{df['T2M_MAX'].mean():.1f} C")
    if "T2M_MIN" in df:
        c3.metric("Average daily low", f"{df['T2M_MIN'].mean():.1f} C")

    fig = go.Figure(go.Bar(x=df.index, y=df["PRECTOTCORR"], marker_color="#2f6f4e"))
    fig.update_layout(yaxis_title="Rain (mm per day)", height=260,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, key="live_rain")
    st.caption(f"Latest day available: {df.index.max():%d %b %Y}. Source: NASA POWER daily data, "
               f"fetched {fetched}. NASA POWER is usually 2 to 3 days behind, and its cells are "
               "about 50 km wide, so neighbouring plots show almost the same weather. "
               "This is real, current weather shown for context. The forecast model does not use it.")
