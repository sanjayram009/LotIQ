"""
LotIQ - QC Manager Dashboard (Streamlit).

Run it from the repo root:
    streamlit run dashboard/app.py

The dashboard reads a pre-scored warehouse snapshot for the overview table and
calls the live model for the "score a new lot" form. All scoring logic is the
same ``lotiq`` library the API uses, so numbers never diverge between them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make `lotiq` importable even if the package wasn't `pip install -e .`-ed.
try:
    from lotiq.config import (
        FEATURE_LABELS,
        FEATURE_UNITS,
        RISK_BANDS,
        SENSOR_FEATURES,
        WAREHOUSE_SNAPSHOT_CSV,
        band_for_score,
    )
    from lotiq.models.predict import load_bundle, score_record
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from lotiq.config import (
        FEATURE_LABELS,
        FEATURE_UNITS,
        RISK_BANDS,
        SENSOR_FEATURES,
        WAREHOUSE_SNAPSHOT_CSV,
        band_for_score,
    )
    from lotiq.models.predict import load_bundle, score_record


st.set_page_config(page_title="LotIQ - QC Dashboard", page_icon="\U0001F336", layout="wide")

_BAND_COLOUR = {b.label: b.colour for b in RISK_BANDS}


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_snapshot() -> pd.DataFrame | None:
    if not Path(WAREHOUSE_SNAPSHOT_CSV).exists():
        return None
    return pd.read_csv(WAREHOUSE_SNAPSHOT_CSV)


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_bundle()


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def risk_gauge(score: float, colour: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": colour},
                "steps": [
                    {"range": [b.lower, min(b.upper, 100)], "color": b.colour + "33"}
                    for b in RISK_BANDS
                ],
                "threshold": {
                    "line": {"color": colour, "width": 4},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def contribution_bar(contributions: list[dict]) -> go.Figure:
    """Horizontal bar of signed contributions (red = raises risk, green = lowers)."""
    ordered = sorted(contributions, key=lambda c: c["contribution"])
    labels = [c["label"] for c in ordered]
    values = [c["contribution"] for c in ordered]
    colours = ["#d13438" if v > 0 else "#2e9e5b" for v in values]

    fig = go.Figure(
        go.Bar(x=values, y=labels, orientation="h", marker_color=colours,
               text=[f"{v:+.1f}" for v in values], textposition="outside")
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Impact on risk score (points)",
        yaxis_title="",
    )
    return fig


def simulated_trend(lot_id: str, current_score: float, days: int = 14) -> go.Figure:
    """A plausible risk trajectory ending at the current score.

    Deterministic per lot (seeded from the lot id) so the same lot always shows
    the same history. This stands in for a real time series until sensor history
    is wired up; it is illustrative, not a model output.
    """
    seed = int(hashlib.md5(lot_id.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    # Start lower and drift up to the current score, with mild noise.
    start = max(0.0, current_score - rng.uniform(15, 40))
    trajectory = np.linspace(start, current_score, days) + rng.normal(0, 2.5, days)
    trajectory = np.clip(trajectory, 0, 100)
    trajectory[-1] = current_score
    x = [f"D-{days - 1 - i}" for i in range(days)]

    fig = go.Figure(go.Scatter(x=x, y=trajectory, mode="lines+markers",
                               line=dict(color="#3b6fb0", width=3)))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(range=[0, 100], title="Risk score"),
        xaxis_title="Day (relative to now)",
    )
    return fig


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Warehouse health")

    counts = df["risk_level"].value_counts()
    c0, c1, c2, c3, c4 = st.columns(5)
    c0.metric("Total lots", len(df))
    c1.metric("\U0001F534 Critical", int(counts.get("Critical", 0)))
    c2.metric("\U0001F7E0 High", int(counts.get("High", 0)))
    c3.metric("\U0001F7E1 Moderate", int(counts.get("Moderate", 0)))
    c4.metric("\U0001F7E2 Healthy", int(counts.get("Healthy", 0)))

    st.markdown("---")

    levels = st.multiselect(
        "Filter by risk level",
        options=["Critical", "High", "Moderate", "Healthy"],
        default=["Critical", "High", "Moderate", "Healthy"],
    )
    view = df[df["risk_level"].isin(levels)].copy()

    display = view[[
        "lot_id", "commodity", "risk_score", "risk_level",
        "moisture_content", "temperature", "humidity", "storage_days", "top_factor_1",
    ]].rename(columns={
        "lot_id": "Lot", "commodity": "Commodity", "risk_score": "Risk",
        "risk_level": "Status", "moisture_content": "Moisture %",
        "temperature": "Temp \u00b0C", "humidity": "Humidity %",
        "storage_days": "Days", "top_factor_1": "Top factor",
    })

    def _colour_status(val: str) -> str:
        return f"background-color: {_BAND_COLOUR.get(val, '#ffffff')}33"

    # pandas >=2.1 uses Styler.map; older versions use applymap.
    styler = display.style
    _apply = getattr(styler, "map", None) or styler.applymap
    styled = _apply(_colour_status, subset=["Status"]).format(
        {"Risk": "{:.0f}", "Moisture %": "{:.1f}", "Temp \u00b0C": "{:.1f}",
         "Humidity %": "{:.0f}", "Days": "{:.0f}"}
    )
    st.dataframe(styled, use_container_width=True, height=430)

    st.caption(
        "Sorted by risk. Select a lot below to see its detail, drivers and "
        "recommended action."
    )

    lot_id = st.selectbox("Inspect a lot", options=view["lot_id"].tolist())
    if lot_id:
        render_lot_detail(df[df["lot_id"] == lot_id].iloc[0])


def render_lot_detail(lot: pd.Series) -> None:
    st.markdown(f"### {lot['emoji']} Lot {lot['lot_id']} - {lot['risk_level']}")

    left, right = st.columns([1, 1])
    with left:
        band = band_for_score(float(lot["risk_score"]))
        st.plotly_chart(risk_gauge(float(lot["risk_score"]), band.colour),
                        use_container_width=True)
        st.info(f"**Recommended action:** {lot['recommendation']}")

    with right:
        st.markdown("**Sensor & lab readings**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Moisture", f"{lot['moisture_content']:.1f}%")
        m2.metric("Temperature", f"{lot['temperature']:.1f}\u00b0C")
        m3.metric("Humidity", f"{lot['humidity']:.0f}%")
        m4, m5, m6 = st.columns(3)
        m4.metric("CO2", f"{lot['co2_level']:.0f} ppm")
        m5.metric("Oleoresin", f"{lot['oleoresin_content']:.1f}%")
        m6.metric("Colour", f"{lot['colour_score']:.0f}/100")

        st.markdown("**Top risk factors**")
        factors = [lot.get(f"top_factor_{i}", "") for i in (1, 2, 3)]
        factors = [f for f in factors if isinstance(f, str) and f]
        if factors:
            for f in factors:
                st.markdown(f"- {f}")
        else:
            st.markdown("- No individual factor stands out; the lot looks stable.")

    st.markdown("**Why this score?**")
    st.write(lot["explanation"])

    t1, t2 = st.columns([1, 1])
    with t1:
        st.markdown("**Risk trend (illustrative)**")
        st.plotly_chart(simulated_trend(lot["lot_id"], float(lot["risk_score"])),
                        use_container_width=True)
    with t2:
        # Re-score live to get full signed contributions for the bar chart.
        record = {k: lot[k] for k in [
            "lot_id", "commodity", "temperature", "humidity", "moisture_content",
            "co2_level", "oleoresin_content", "colour_score", "storage_days",
            "initial_quality_grade",
        ] if k in lot}
        scored = score_record(record, bundle=get_bundle())
        st.markdown("**Feature contributions**")
        st.plotly_chart(contribution_bar(scored["contributions"]),
                        use_container_width=True)


def render_new_lot_form() -> None:
    st.subheader("Score a new lot")
    st.caption("Enter readings and the live model returns a risk score and explanation.")

    c1, c2, c3 = st.columns(3)
    with c1:
        temperature = st.slider("Temperature (\u00b0C)", 12.0, 42.0, 27.0, 0.1)
        humidity = st.slider("Humidity (%)", 30.0, 95.0, 62.0, 0.5)
        moisture = st.slider("Moisture content (%)", 6.0, 20.0, 10.5, 0.1)
    with c2:
        co2 = st.slider("CO2 (ppm)", 400.0, 2500.0, 800.0, 10.0)
        oleoresin = st.slider("Oleoresin content (%)", 2.0, 12.0, 7.0, 0.1)
        colour = st.slider("Colour score (/100)", 20.0, 100.0, 78.0, 1.0)
    with c3:
        storage_days = st.slider("Storage days", 0, 120, 30, 1)
        grade = st.selectbox("Initial quality grade", ["A", "B", "C"], index=1)
        lot_id = st.text_input("Lot ID", value="L-NEW")

    if st.button("Score lot", type="primary"):
        record = {
            "lot_id": lot_id, "commodity": "chilli", "temperature": temperature,
            "humidity": humidity, "moisture_content": moisture, "co2_level": co2,
            "oleoresin_content": oleoresin, "colour_score": colour,
            "storage_days": storage_days, "initial_quality_grade": grade,
        }
        scored = score_record(record, bundle=get_bundle())
        band = band_for_score(scored["risk_score"])

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(risk_gauge(scored["risk_score"], band.colour),
                            use_container_width=True)
            st.markdown(f"### {scored['emoji']} {scored['risk_level']}")
            st.info(f"**Recommended action:** {scored['recommendation']}")
        with right:
            st.markdown("**Why this score?**")
            st.write(scored["explanation"])
            if scored["risk_factors"]:
                st.markdown("**Top risk factors**")
                for f in scored["risk_factors"]:
                    st.markdown(f"- {f}")
            if scored["protective_factors"]:
                st.markdown("**Protective factors**")
                for f in scored["protective_factors"]:
                    st.markdown(f"- {f}")
        st.plotly_chart(contribution_bar(scored["contributions"]),
                        use_container_width=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("\U0001F336\uFE0F LotIQ - Spice Warehouse QC Dashboard")
    st.caption(
        "Prototype - trained on SYNTHETIC data, not real operational data. "
        "Pilot commodity: chilli."
    )

    df = load_snapshot()

    tab1, tab2 = st.tabs(["\U0001F4CA Warehouse", "\u2795 Score a new lot"])
    with tab1:
        if df is None:
            st.warning(
                "No warehouse snapshot found. Generate one first:\n\n"
                "```\npython -m lotiq.data.snapshot\n```"
            )
        else:
            render_overview(df)
    with tab2:
        try:
            get_bundle()
            render_new_lot_form()
        except FileNotFoundError:
            st.warning(
                "No trained model found. Train one first:\n\n"
                "```\npython -m lotiq.models.train\n```"
            )


if __name__ == "__main__":
    main()
