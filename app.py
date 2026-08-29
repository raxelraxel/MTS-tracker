import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pandas as pd
from pathlib import Path
import numpy as np


#Format changes to change "g" in Plotly to "b"; Plotly uses d3 which automatically does G for billions.
#These helper formatting functions fix that. Sorry its long.
def format_currency_bt(value: float) -> str:
    """Format a dollar value using B/T/M/K suffixes instead of d3's G/M/k."""
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}${abs_val / 1e12:.2f}T"
    elif abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.2f}M"
    elif abs_val >= 1e3:
        return f"{sign}${abs_val / 1e3:.1f}K"
    else:
        return f"{sign}${abs_val:.2f}"

def _nice_step(raw_step: float) -> float:
    """Round a raw tick step up to a 'nice' 1/2/5 x 10^n value."""
    if raw_step <= 0:
        return 1
    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / 10**exponent
    nice_fraction = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    return nice_fraction * 10**exponent

def build_nice_ticks(vmin: float, vmax: float, n_ticks: int = 5):
    """Return (tickvals, ticktext) as round numbers spanning [vmin, vmax]."""
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    step = _nice_step((vmax - vmin) / max(n_ticks - 1, 1))
    start = np.floor(vmin / step) * step
    ticks = []
    t = start
    for _ in range(n_ticks + 4):
        ticks.append(t)
        if t > vmax:
            break
        t += step
    return ticks, [format_currency_bt(t) for t in ticks]


@st.cache_data(ttl=3600)  # cache for 1 hour so repeated interactions are fast
def load_mts_data() -> pd.DataFrame:
    df = pd.read_csv(Path("data/mts_data.csv"))
    df["record_date"] = pd.to_datetime(df["record_date"])
    return df

@st.cache_data
def load_dictionary() -> pd.DataFrame:
    return pd.read_csv(Path("data/MTS dictionary.csv"))

dict_df = load_dictionary()
df = load_mts_data()
var_descriptions = dict_df.set_index("variable")["description"].to_dict()


@st.cache_data
def load_groq_summaries() -> list:
    with open(Path("data/groq_summaries.json"), "r", encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        r["record_date"] = pd.to_datetime(r["record_date"]).strftime("%Y-%m-%d")
    return records

@st.cache_data
def load_anomaly_flags() -> pd.DataFrame:
    df = pd.read_csv(Path("data/output_anomaly_flags.csv"))
    df["record_date"] = pd.to_datetime(df["record_date"], format="mixed").dt.strftime("%Y-%m-%d")
    return df

summaries = load_groq_summaries()
anomaly_flags = load_anomaly_flags()

record_type_map = {
    "Raw": "raw",
    "Standardized (z-score)": "zscore",
    "Real": "real",
    "De-seasonalized": "deseasonalized",
}


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MTS Dashboard (DRAFT - NOT FOR RELEASE)",
    page_icon="📊",
    layout="wide",
)

# ── App header ─────────────────────────────────────────────────────────────────
st.title("📊 Monthly Treasury Statement Dashboard")
st.caption("U.S. Treasury — Monthly Financial Data Explorer")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Selectors
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ══════════════════════════════════════════════════════
    # DATA MODE
    # ══════════════════════════════════════════════════════
    st.header("Data Formatting")
    data_mode = st.radio(
        "Format:",
        options=["Raw", "Standardized (z-score)", "Real", "De-seasonalized"],
        horizontal=True,
    )

    st.markdown("---")

    # ══════════════════════════════════════════════════════
    # ANOMALY DETECTION
    # ══════════════════════════════════════════════════════
    st.header("Anomaly Detection")

    available_dates = sorted(
        [s["record_date"] for s in summaries],
        reverse=True
    )
    date_labels = {s: pd.to_datetime(s).strftime("%B %Y") for s in available_dates}

    selected_summary_date = st.selectbox(
        "Select month:",
        options=available_dates,
        format_func=lambda d: date_labels[d],
        index=0
    )

    show_summary = st.toggle("View anomaly results", value=False)

    st.markdown("---")

    # ══════════════════════════════════════════════════════
    # VARIABLES VIEWED
    # ══════════════════════════════════════════════════════
    st.header("Variables Viewed")

    # ── Filter mode ───────────────────────────────────────
    filter_mode = st.radio(
        "Filter by:",
        options=["Group", "MTS Table","Anomalies"],
        horizontal=True,
    )

    if filter_mode == "Group":
        groups = dict_df.loc[dict_df["group"] != "metadata", "group"].unique().tolist() #Exclude "metadata" from the drop down
        selected_filter = st.selectbox(
            "Select Group",
            options=groups,
            index=groups.index("summary"),
        )
        filtered_vars = dict_df[dict_df["group"] == selected_filter]["variable"].tolist()
    elif filter_mode=="MTS Table":
        tables = sorted(dict_df["mts_table"].unique().tolist(), key=str)
        selected_filter = st.selectbox(
            "Select MTS Table",
            options=tables,
            format_func=lambda x: f"Table {x}",
        )
        filtered_vars = dict_df[dict_df["mts_table"] == selected_filter]["variable"].tolist()
    else: #filter_mode=="Anomalies"
        filtered_vars = anomaly_flags[
            pd.to_datetime(anomaly_flags["record_date"]) == pd.to_datetime(selected_summary_date)
            ]["variable"].tolist()
        if not filtered_vars:
            st.caption("No anomalies flagged for this month.")
    filtered_vars = list(dict.fromkeys(filtered_vars))
    st.markdown("---")

    # ── Variable checkboxes ───────────────────────────────
    # ── Variable checkboxes ───────────────────────────────
    st.subheader("Variables")

    default_selected = set(filtered_vars[:6])  # pre-check only the first 6 by default

    selected_vars = []
    for var in filtered_vars:
        checked = st.checkbox(
            var,
            value=(var in default_selected),
            key=f"cb_{var}",
            help=var_descriptions.get(var, "No description available."),
        )
        if checked:
            selected_vars.append(var)

    if len(selected_vars) > 6:
        all_cols = [c for c in df.columns if c in selected_vars]
        selected_vars = all_cols[:6]
        st.caption("⚠️ Max 6 variables. Showing first 6 checked.")

# ── Determine variables to plot ───────────────────────────────────────────────
chart_vars = selected_vars

# ══════════════════════════════════════════════════════════════════════════════
# Groq narrative summary
# ══════════════════════════════════════════════════════════════════════════════
if show_summary:
    st.markdown("### Potential Anomalies for Selected Month")
    selected_summary = next(
        (s["summary"] for s in summaries if s["record_date"] == selected_summary_date),
        "No summary available for this date."
    )
    selected_summary = selected_summary.replace("$", "\\$")
    st.info(selected_summary)

# ══════════════════════════════════════════════════════════════════════════════
# DATE RANGE SLIDER
# ══════════════════════════════════════════════════════════════════════════════
min_date = df["record_date"].min()
max_date = df["record_date"].max()

# Convert to plain Python dates for the slider (Streamlit requires this)
min_d = min_date.date()
max_d = max_date.date()

st.markdown("### Date Range")
date_range = st.slider(
    label="Select date range",
    min_value=min_d,
    max_value=max_d,
    value=(min_d, max_d),        # default: full range selected
    format="MMM YYYY",           # display as e.g. "Jan 2021"
)

# Filter the dataframe to the selected date range
start_date, end_date = date_range
df_filtered = df[
    (df["record_type"] == record_type_map[data_mode]) &
    (df["record_date"].dt.date >= start_date) &
    (df["record_date"].dt.date <= end_date)
].copy()

# ══════════════════════════════════════════════════════════════════════════════
# FACETED LINE CHARTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Chart title reflecting active mode and date range ─────────────────────────
is_standardized = data_mode == "Standardized (z-score)"
mode_label = {
    "Raw": "USD (Nominal)",
    "Standardized (z-score)": "Z-Score",
    "Real": "USD (Real)",
    "De-seasonalized": "USD (De-seasonalized)",
}[data_mode]
chart_title = (
    f"MTS Data — {start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')}  "
    f"| {mode_label}"
)
st.markdown(f"### {chart_title}")

if not chart_vars:
    st.warning("No variables selected. Please select at least one variable.")

elif df_filtered.empty:
    st.warning("No data in the selected date range.")

else:
    n = len(chart_vars)
    # Build subplot grid — 2 columns, enough rows to fit all charts
    n_cols = 2
    n_rows = (n + 1) // 2   # ceiling division: e.g. 5 vars → 3 rows

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=chart_vars,
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )

    # ── Color palette — one color per variable ────────────────────────────────
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b",
    ]

    # ── Plot one line per variable in its own facet ───────────────────────────
    for i, var in enumerate(chart_vars):
        row = i // n_cols + 1
        col = i % n_cols + 1

        y_values = df_filtered[var]
        if is_standardized:
            hover_customdata = None
            hover_val_part = f"{var}: %{{y:.2f}}σ"
        else:
            hover_customdata = [format_currency_bt(v) for v in y_values]
            hover_val_part = f"{var}: %{{customdata}}"

        fig.add_trace(
            go.Scatter(
                x=df_filtered["record_date"],
                y=y_values,
                mode="lines+markers",
                name=var,
                line=dict(width=2, color=colors[i % len(colors)]),
                marker=dict(size=4, color=colors[i % len(colors)]),
                showlegend=False,
                customdata=hover_customdata,
                hovertemplate=(
                        "<b>%{x|%b %Y}</b><br>"
                        + hover_val_part
                        + "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    # ── Y-axis formatting: dollars for raw, z-score for standardized ──────────
    for i, var in enumerate(chart_vars):
        axis_key = f"yaxis{i + 1 if i > 0 else ''}"
        if is_standardized:
            fig.update_layout(**{
                axis_key: dict(tickformat=".1f", ticksuffix="σ", zeroline=True,
                               zerolinecolor="#cccccc", zerolinewidth=1)
            })
        else:
            vmin = df_filtered[var].min()
            vmax = df_filtered[var].max()
            tickvals, ticktext = build_nice_ticks(vmin, vmax)
            fig.update_layout(**{
                axis_key: dict(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            })

    # ── Subplot title font ────────────────────────────────────────────────────
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=11, color="#444444")

    # ── Overall figure layout ─────────────────────────────────────────────────
    chart_height = 320 * n_rows

    fig.update_layout(
        height=chart_height,
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        margin=dict(t=40, b=40, l=60, r=20),
        font=dict(family="Arial, sans-serif", size=11),
    )

    # Light gridlines and clean x-axis date formatting
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#eeeeee",
        tickformat="%b %Y",
        tickangle=-30,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#eeeeee",
    )

    st.plotly_chart(fig, width="stretch")

    # ── Footer note ───────────────────────────────────────────────────────────
    if is_standardized:
        st.caption("Values shown as z-scores (mean=0, sd=1). Each variable standardized independently across the full dataset.")
    else:
        st.caption("Values shown in USD. Source: U.S. Treasury Monthly Treasury Statement (MTS).")