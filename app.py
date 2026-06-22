import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pandas as pd
from pathlib import Path

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
    # Build a sorted list of available dates from the JSON for anomaly detection
    st.header("Select Date for Anomaly Detection")
    available_dates = sorted(
        [s["record_date"] for s in summaries],
        reverse=True
    )

    # Format for display — strip the timestamp portion
    date_labels = {s: pd.to_datetime(s).strftime("%B %Y") for s in available_dates}

    selected_summary_date = st.selectbox(
        "Select month:",
        options=available_dates,
        format_func=lambda d: date_labels[d],
        index=0  # default to most recent
    )

    st.header("Select Variables")

    # ── Data mode: Raw or Standardized ────────────────────────────────────────
    st.markdown("---")
    data_mode = st.radio(
        "Data:",
        options=["Raw", "Standardized (z-score)","Real","De-seasonalized"],
        horizontal=True,
    )
    st.markdown("---")

    # ── Filter mode: Group or MTS Table ───────────────────────────────────────
    filter_mode = st.radio(
        "Filter by:",
        options=["Group", "MTS Table"],
        horizontal=True,
    )

    # ── Group dropdown ─────────────────────────────────────────────────────────
    if filter_mode == "Group":
        groups = dict_df["group"].unique().tolist()
        selected_filter = st.selectbox(
            "Select Group",
            options=groups,
            index=groups.index("summary"),   # default to summary
        )
        filtered_vars = dict_df[dict_df["group"] == selected_filter]["variable"].tolist()

    # ── MTS Table dropdown ─────────────────────────────────────────────────────
    else:
        tables = sorted(dict_df["mts_table"].unique().tolist(), key=str)
        selected_filter = st.selectbox(
            "Select MTS Table",
            options=tables,
            format_func=lambda x: f"Table {x}",
        )
        filtered_vars = dict_df[dict_df["mts_table"] == selected_filter]["variable"].tolist()

    # ── Variable checkboxes with Select All ───────────────────────────────────
    st.markdown("---")
    st.subheader("Variables")

    # REMOVED FOR NOW: Select All toggle
    #select_all = st.checkbox("Select All", value=True)

    # ── Anomalies filter ──────────────────────────────────────────────────────────
    show_anomalies = st.checkbox("Anomalies only", value=False)

    if show_anomalies:
        # Use the same date selected in the summary dropdown
        anomaly_vars = anomaly_flags[
            pd.to_datetime(anomaly_flags["record_date"]) == pd.to_datetime(selected_summary_date,format="mixed")
            ]["variable"].tolist()
        filtered_vars = [v for v in filtered_vars if v in anomaly_vars]
        if not filtered_vars:
            st.caption("No anomalies flagged for the current filter selection.")

    # If Select All is checked and there are more than 6 variables,
    # pre-select only the first 6 in data column order before rendering
    if len(filtered_vars) > 6:
        all_cols = [c for c in df.columns if c in filtered_vars]
        default_selected = set(all_cols[:6])
        st.caption("⚠️ Max 6 variables shown. First 6 pre-selected.")
    else:
        default_selected = set(filtered_vars)

    # Render one checkbox per variable — checked state reflects the 6-var limit
    selected_vars = []
    for var in filtered_vars:
        checked = st.checkbox(var, value=(var in default_selected), key=f"cb_{var}")
        if checked:
            selected_vars.append(var)

    # Final safety cap — if user manually checks more than 6
    if len(selected_vars) > 6:
        all_cols = [c for c in df.columns if c in selected_vars]
        selected_vars = all_cols[:6]
        st.caption("⚠️ Max 6 variables. Showing first 6 checked.")

# ══════════════════════════════════════════════════════════════════════════════
# Groq narrative summary
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### Analysis Summary")
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

if not selected_vars:
    st.warning("No variables selected. Please select at least one variable in the sidebar.")

elif df_filtered.empty:
    st.warning("No data in the selected date range.")

else:
    n = len(selected_vars)

    # Build subplot grid — 2 columns, enough rows to fit all charts
    n_cols = 2
    n_rows = (n + 1) // 2   # ceiling division: e.g. 5 vars → 3 rows

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=selected_vars,
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )

    # ── Color palette — one color per variable ────────────────────────────────
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b",
    ]

    # ── Plot one line per variable in its own facet ───────────────────────────
    for i, var in enumerate(selected_vars):
        row = i // n_cols + 1
        col = i % n_cols + 1

        fig.add_trace(
            go.Scatter(
                x=df_filtered["record_date"],
                y=df_filtered[var],
                mode="lines+markers",
                name=var,
                line=dict(width=2, color=colors[i % len(colors)]),
                marker=dict(size=4, color=colors[i % len(colors)]),
                showlegend=False,
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    + (f"{var}: %{{y:$.3s}}" if not is_standardized else f"{var}: %{{y:.2f}}σ")
                    + "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    # ── Y-axis formatting: dollars for raw, z-score for standardized ──────────
    for i in range(1, n + 1):
        axis_key = f"yaxis{i if i > 1 else ''}"
        if is_standardized:
            # Plain decimal for z-scores, with σ suffix
            fig.update_layout(**{
                axis_key: dict(tickformat=".1f", ticksuffix="σ", zeroline=True, zerolinecolor="#cccccc", zerolinewidth=1)
            })
        else:
            # Dollar billions/trillions
            fig.update_layout(**{
                axis_key: dict(tickformat="$.3s")
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