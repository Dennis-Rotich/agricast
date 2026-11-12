import plotly.graph_objects as go
from src.advisory.rule_engine import CHART_LABEL


def fan_chart(cell):
    """cell: the two rows (14d and 28d) for one grid cell and forecast date."""
    x = cell.horizon_days
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=cell.vci_q90, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=cell.vci_q10, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(47,111,78,0.2)",
                             name="Likely range"))
    fig.add_trace(go.Scatter(x=x, y=cell.vci_q50, mode="lines+markers",
                             line=dict(color="#2f6f4e"), name="Most likely value"))
    if "vci_actual" in cell and cell.vci_actual.notna().any():
        fig.add_trace(go.Scatter(x=x, y=cell.vci_actual, mode="markers",
                                 marker=dict(color="#1f2a24", size=9, symbol="diamond"),
                                 name="What actually happened"))
    fig.add_hline(y=35, line_dash="dash", line_color="#c0392b",
                  annotation_text="Drought stress line (VCI 35)")
    fig.update_layout(
        xaxis=dict(title="Days ahead", tickvals=[14, 28]),
        yaxis=dict(title="Crop health (VCI)", range=[0, 100]),
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


def shap_chart(row):
    items = [(CHART_LABEL.get(row[f"shap_top{k}_feature"], row[f"shap_top{k}_feature"]),
              row[f"shap_top{k}_val"]) for k in (1, 2, 3)]
    items.reverse()  # plotly draws the first bar at the bottom
    labels = [n for n, _ in items]
    vals = [v for _, v in items]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=["#c0392b" if v > 0 else "#2f8f5b" for v in vals],
        text=[f"{v:+.1f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="What is driving this forecast", font=dict(size=14)),
        xaxis=dict(title="Effect on risk (VCI points)", zeroline=True),
        height=220, margin=dict(l=0, r=30, t=40, b=0),
    )
    return fig
