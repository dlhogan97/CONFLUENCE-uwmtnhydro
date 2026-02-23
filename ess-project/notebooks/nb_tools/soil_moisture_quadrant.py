"""
Generated using Claude Sonnet 4.6

Fall Soil Moisture Quadrant Plot — Plotly / Jupyter version
------------------------------------------------------------
Dependencies:
    pip install plotly pandas numpy ipywidgets

No special backend setup needed. Just call:
    plot_soil_moisture_quadrant(df)

df must have columns:
    precip        – precipitation anomaly (z-score / σ)
    temp          – temperature anomaly (z-score / σ)
    soil_moisture – fall soil moisture (σ)
    year          – year (used in hover labels)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display


# ── Colorscale: sandy tan (dry) → deep blue (wet) ───────────────────────────

SM_COLORSCALE = [
    [0.0,  "#c8935a"],
    [0.4,  "#a0845c"],
    [0.6,  "#6e8fa8"],
    [1.0,  "#1e4e8c"],
]

# Quadrant config: (x_range, y_range, fill_color, label, label_xy)
QUADRANTS = [
    ([-3, 0], [0,  3], "rgba(239,68,68,0.07)",   "Warm & Dry", (-2,  2.5)),
    ([ 0, 3], [0,  3], "rgba(249,115,22,0.07)",  "Warm & Wet", ( 1,  2.5)),
    ([-3, 0], [-3, 0], "rgba(148,163,184,0.07)", "Cool & Dry", (-2, -2.50)),
    ([ 0, 3], [-3, 0], "rgba(96,165,250,0.07)",  "Cool & Wet", ( 1, -2.50)),
]


def _build_figure(df, sm_min, sm_max):
    """Build the base figure with quadrant shading and reference lines."""
    fig = go.Figure()

    # Quadrant shaded rectangles
    for (xr, yr, color, label, lxy) in QUADRANTS:
        fig.add_shape(
            type="rect",
            x0=xr[0], x1=xr[1], y0=yr[0], y1=yr[1],
            fillcolor=color,
            line=dict(color="rgba(0,0,0,0)"),
            layer="below",
        )
        fig.add_annotation(
            x=lxy[0], y=lxy[1],
            text=label,
            showarrow=False,
            font=dict(size=16, color="rgba(200,200,200,0.6)"),
        )

    # Reference lines
    fig.add_hline(y=0, line=dict(color="#334155", width=2, dash="dash"))
    fig.add_vline(x=0, line=dict(color="#334155", width=2, dash="dash"))

    # Update layout with larger, brighter text
    fig.update_layout(
        xaxis=dict(
            title=dict(text="JAS Precipitation Anomaly (σ)", font=dict(size=20, color="#ffffff")),
            tickfont=dict(size=18, color="#ffffff"),
            showgrid=True,
            gridwidth=1,
            range=[-3, 3],
            gridcolor="rgba(100,100,100,0.2)",
        ),
        yaxis=dict(
            title=dict(text="Sept. Temperature Anomaly (σ)", font=dict(size=20, color="#ffffff")),
            tickfont=dict(size=18, color="#ffffff"),
            showgrid=True,
            gridwidth=1,
            range=[-3, 3],
            gridcolor="rgba(100,100,100,0.2)",
        ),
        plot_bgcolor="#0f1419",
        paper_bgcolor="#0f1419",
        font=dict(family="Arial, sans-serif", size=14, color="#f0f0f0"),
        hovermode="closest",
        showlegend=False,
        margin=dict(l=80, r=100, t=80, b=80),
    )

    return fig


def _scatter_trace(sub, sm_min, sm_max):
    """Create a scatter trace for a subset of the data."""
    return go.Scatter(
        x=sub["precip"],
        y=sub["temp"],
        mode="markers",
        marker=dict(
            color=sub["soil_moisture"],
            colorscale=SM_COLORSCALE,
            cmin=sm_min,
            cmax=sm_max,
            size=9,
            opacity=0.88,
            line=dict(color="white", width=0.4),
            colorbar=dict(
                title=dict(text="Oct 1 Soil <br>Moisture [8-in] (σ)",
                           font=dict(color="#f0f0f0", size=14)),
                tickfont=dict(color="#e0e0e0", size=12),
                outlinecolor="#2d3548",
                outlinewidth=1,
                thickness=20,
                len=0.85,
            ),
        ),
        customdata=np.stack([sub["water_year"], sub["soil_moisture"]], axis=1),
        hovertemplate=(
            "<b>%{customdata[0]:.0f}</b><br>"
            "Precip: %{x:+.2f} σ<br>"
            "Temp: %{y:+.2f} σ<br>"
            "SM: %{customdata[1]:.1f}σ"
            "<extra></extra>"
        ),
    )


def plot_soil_moisture_quadrant(df: pd.DataFrame, sm_min=None, sm_max=None):
    """
    Interactive quadrant plot of fall soil moisture vs climate anomalies.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
            precip        – precipitation anomaly (z-score / σ)
            temp          – temperature anomaly (z-score / σ)
            soil_moisture – fall soil moisture (σ)
            year          – year (used in hover labels)
    sm_min : float, optional
        Minimum soil moisture value for colorscale (default: min of data)
    sm_max : float, optional
        Maximum soil moisture value for colorscale (default: max of data)
    """
    df = df.copy().reset_index(drop=True)
    
    # Use provided values or calculate from data
    if sm_min is None:
        sm_min = df["soil_moisture"].min()
    if sm_max is None:
        sm_max = df["soil_moisture"].max()

    # ── Widgets ──────────────────────────────────────────────────────────────

    sl_lo = widgets.FloatSlider(
        value=-1, min=-2, max=-0.5, step=0.1,
        description="Dry ≤",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="300px"),
    )
    sl_hi = widgets.FloatSlider(
        value=1, min=0.5, max=2, step=0.1,
        description="Wet ≥",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="300px"),
    )
    toggle = widgets.ToggleButtons(
        options=["All years", "Dry SM", "Mid SM", "Wet SM"],
        value="All years",
        description="Show:",
        style={"description_width": "50px", "button_width": "88px"},
    )
    count_label = widgets.Label(value=f"n = {len(df)} / {len(df)} years shown")

    # ── Initial figure ───────────────────────────────────────────────────────

    fig = _build_figure(df, sm_min, sm_max)
    fig.add_trace(_scatter_trace(df, sm_min, sm_max))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1320",
        plot_bgcolor="#12172a",
        title=dict(
            text="Fall Soil Moisture at Butte SNOTEL Climate Quadrant Plot",
            font=dict(color="#e2e8f0", size=16),
            x=0.5,
        ),
        margin=dict(l=60, r=80, t=60, b=60),
        height=560,
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#1a1f2e",
            bordercolor="#2d3548",
            font=dict(color="#e2e8f0", size=11, family="monospace"),
        ),
    )

    fw = go.FigureWidget(fig)
    scatter = fw.data[0]

    # ── Update callback ───────────────────────────────────────────────────────

    def update(change=None):
        lo   = sl_lo.value
        hi   = sl_hi.value
        mode = toggle.value

        if mode == "Dry SM":
            mask = df["soil_moisture"] <= lo
        elif mode == "Wet SM":
            mask = df["soil_moisture"] >= hi
        elif mode == "Mid SM":
            mask = (df["soil_moisture"] > lo) & (df["soil_moisture"] < hi)
        else:
            mask = pd.Series([True] * len(df), index=df.index)

        sub = df[mask]

        with fw.batch_update():
            scatter.x = sub["precip"]
            scatter.y = sub["temp"]
            scatter.marker.color = sub["soil_moisture"]
            scatter.marker.size  = 11 if mode != "All years" else 9
            scatter.customdata   = np.stack(
                [sub["water_year"], sub["soil_moisture"]], axis=1
            )

        count_label.value = f"n = {len(sub)} / {len(df)} years shown"

    sl_lo.observe(update, names="value")
    sl_hi.observe(update, names="value")
    toggle.observe(update, names="value")

    # ── Layout & display ──────────────────────────────────────────────────────

    ui = widgets.VBox([
        widgets.HBox([toggle, count_label],
                     layout=widgets.Layout(align_items="center", gap="20px")),
        widgets.HBox([sl_lo, sl_hi]),
        fw,
    ])

    display(ui)