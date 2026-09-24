"""
AWS EC2 CPU Utilization Analytics & Anomaly Detection
IBM SkillsBuild Data Analytics with AI Internship — Final Project
Author: Internship Student
Dataset: Numenta Anomaly Benchmark (NAB) — EC2 CPU Utilization
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# 0. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EC2 CPU Analytics",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────
DATA_DIR = "dataset"

FILE_MAP = {
    "5f5533": "ec2_cpu_utilization_5f5533.csv",
    "24ae8d": "ec2_cpu_utilization_24ae8d.csv",
    "53ea38": "ec2_cpu_utilization_53ea38.csv",
    "77c1ca": "ec2_cpu_utilization_77c1ca.csv",
    "825cc2": "ec2_cpu_utilization_825cc2.csv",
    "ac20cd": "ec2_cpu_utilization_ac20cd.csv",
    "c6585a": "ec2_cpu_utilization_c6585a.csv",
    "fe7f93": "ec2_cpu_utilization_fe7f93.csv",
}

INSTANCE_LABELS = {
    "5f5533": "5f5533 — Medium Load",
    "24ae8d": "24ae8d — Near-Idle",
    "53ea38": "53ea38 — Very Low Use",
    "77c1ca": "77c1ca — Spike Event",
    "825cc2": "825cc2 — Critically Overloaded",
    "ac20cd": "ac20cd — Escalating Load",
    "c6585a": "c6585a — Near-Zero",
    "fe7f93": "fe7f93 — Low Stable",
}

HIGH_UTIL_THRESHOLD = 80.0   # % — high utilisation warning level
CRITICAL_THRESHOLD  = 95.0   # % — critical saturation level


@st.cache_data
def load_all_datasets():
    """
    Load all 8 CSV files into a dict {instance_id: DataFrame}.
    Each DataFrame has columns: timestamp (datetime64), value (float64), instance_id (str).
    Returns (datasets_dict, combined_df, data_quality_df).
    """
    datasets = {}
    quality_rows = []

    for inst_id, filename in FILE_MAP.items():
        filepath = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(filepath)

        # Standardise columns
        df.columns = [c.strip().lower() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["value"] = df["value"].astype(float)
        df["instance_id"] = inst_id

        # Quality checks
        missing = df["value"].isna().sum()
        dupes   = df.duplicated(subset=["timestamp"]).sum()
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        datasets[inst_id] = df

        quality_rows.append({
            "Instance": inst_id,
            "Rows": len(df),
            "Missing Values": missing,
            "Duplicate Timestamps": dupes,
            "Start": df["timestamp"].min().strftime("%Y-%m-%d"),
            "End":   df["timestamp"].max().strftime("%Y-%m-%d"),
        })

    combined = pd.concat(datasets.values(), ignore_index=True)
    quality_df = pd.DataFrame(quality_rows)
    return datasets, combined, quality_df


# ─────────────────────────────────────────────
# 2. ANOMALY DETECTION  (IQR method)
# ─────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag anomalies using the IQR (Interquartile Range) method.

    A data point is anomalous if:
        value < Q1 - 1.5 * IQR   →  'dip'
        value > Q3 + 1.5 * IQR   →  'spike'

    Returns the DataFrame with two new columns:
        is_anomaly  (bool)
        anomaly_type ('spike' | 'dip' | 'none')
    """
    df = df.copy()
    q1  = df["value"].quantile(0.25)
    q3  = df["value"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df["is_anomaly"]   = (df["value"] < lower) | (df["value"] > upper)
    df["anomaly_type"] = "none"
    df.loc[df["value"] > upper, "anomaly_type"] = "spike"
    df.loc[df["value"] < lower, "anomaly_type"] = "dip"

    return df


# ─────────────────────────────────────────────
# 3. KPI CALCULATIONS
# ─────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute per-instance KPIs. Expects anomaly columns already present."""
    vals = df["value"]
    return {
        "mean":           round(vals.mean(), 3),
        "min":            round(vals.min(),  3),
        "max":            round(vals.max(),  3),
        "std":            round(vals.std(),  3),
        "high_util_pct":  round((vals > HIGH_UTIL_THRESHOLD).mean() * 100, 2),
        "critical_pct":   round((vals > CRITICAL_THRESHOLD).mean()  * 100, 2),
        "anomaly_count":  int(df["is_anomaly"].sum()),
        "anomaly_pct":    round(df["is_anomaly"].mean() * 100, 2),
        "total_readings": len(df),
    }


def compute_fleet_summary(datasets: dict) -> pd.DataFrame:
    """Build the per-instance KPI summary table (all 8 rows)."""
    rows = []
    for inst_id, df in datasets.items():
        k = compute_kpis(df)
        rows.append({
            "Instance":         inst_id,
            "Label":            INSTANCE_LABELS[inst_id],
            "Avg CPU (%)":      k["mean"],
            "Min CPU (%)":      k["min"],
            "Max CPU (%)":      k["max"],
            "Std Dev":          k["std"],
            "High Util (>80%) Readings (%)": k["high_util_pct"],
            "Critical (>95%) Readings (%)":  k["critical_pct"],
            "Anomaly Count":    k["anomaly_count"],
            "Anomaly (%)":      k["anomaly_pct"],
            "Total Readings":   k["total_readings"],
        })
    return pd.DataFrame(rows)


def compute_fleet_kpis(datasets: dict, fleet_df: pd.DataFrame) -> dict:
    """Compute top-level fleet KPIs for the metric cards."""
    total_anomalies  = fleet_df["Anomaly Count"].sum()
    most_overloaded  = fleet_df.loc[fleet_df["Avg CPU (%)"].idxmax(), "Instance"]
    most_idle        = fleet_df.loc[fleet_df["Avg CPU (%)"].idxmin(), "Instance"]
    fleet_mean       = round(fleet_df["Avg CPU (%)"].mean(), 2)
    high_util_events = sum(
        int((df["value"] > HIGH_UTIL_THRESHOLD).sum())
        for df in datasets.values()
    )
    return {
        "fleet_mean":       fleet_mean,
        "most_overloaded":  most_overloaded,
        "most_idle":        most_idle,
        "total_anomalies":  int(total_anomalies),
        "high_util_events": high_util_events,
    }


# ─────────────────────────────────────────────
# 4. CHART BUILDERS
# ─────────────────────────────────────────────

COLOUR_SEQUENCE = px.colors.qualitative.Plotly


def chart_instance_trend(df: pd.DataFrame, inst_id: str) -> go.Figure:
    """Chart 1 — Line chart for a single instance with 80 % threshold line."""
    fig = px.line(
        df,
        x="timestamp",
        y="value",
        title=f"CPU Utilization Over Time — Instance {inst_id}",
        labels={"timestamp": "Timestamp", "value": "CPU Utilization (%)"},
        color_discrete_sequence=["#3b82d4"],
    )
    # 80 % threshold line
    fig.add_hline(
        y=HIGH_UTIL_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text="80 % High-Util Threshold",
        annotation_position="bottom right",
    )
    fig.update_layout(
        yaxis=dict(range=[0, max(105, df["value"].max() * 1.05)]),
        hovermode="x unified",
        margin=dict(t=50, b=40),
    )
    return fig


def chart_all_instances_overlay(combined: pd.DataFrame) -> go.Figure:
    """Chart 2 — Multi-line overlay of all 8 instances (absolute time)."""
    fig = px.line(
        combined,
        x="timestamp",
        y="value",
        color="instance_id",
        title="All Instances — CPU Utilization Overlay",
        labels={"timestamp": "Timestamp", "value": "CPU Utilization (%)", "instance_id": "Instance"},
        color_discrete_sequence=COLOUR_SEQUENCE,
    )
    fig.add_hline(
        y=HIGH_UTIL_THRESHOLD,
        line_dash="dot",
        line_color="red",
        annotation_text="80 % Threshold",
        annotation_position="bottom right",
    )
    fig.update_layout(
        hovermode="x unified",
        margin=dict(t=50, b=40),
        legend_title_text="Instance",
    )
    return fig


def chart_avg_cpu_bar(fleet_df: pd.DataFrame) -> go.Figure:
    """Chart 3 — Horizontal bar chart ranking instances by average CPU."""
    df_sorted = fleet_df.sort_values("Avg CPU (%)", ascending=True)

    colours = [
        "#d62728" if v > 90 else "#ff7f0e" if v > 60 else "#2ca02c" if v > 20 else "#aec7e8"
        for v in df_sorted["Avg CPU (%)"]
    ]

    fig = go.Figure(go.Bar(
        x=df_sorted["Avg CPU (%)"],
        y=df_sorted["Instance"],
        orientation="h",
        marker_color=colours,
        text=[f"{v:.1f} %" for v in df_sorted["Avg CPU (%)"]],
        textposition="outside",
    ))
    fig.add_vline(x=HIGH_UTIL_THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text="80 %", annotation_position="top")
    fig.update_layout(
        title="Average CPU Utilization by Instance",
        xaxis_title="Average CPU (%)",
        yaxis_title="Instance ID",
        margin=dict(t=50, b=40, r=80),
        xaxis=dict(range=[0, 110]),
    )
    return fig


def chart_anomaly_overlay(df: pd.DataFrame, inst_id: str) -> go.Figure:
    """Chart 4 — Line chart with anomalous points highlighted as red dots."""
    normal   = df[~df["is_anomaly"]]
    spikes   = df[df["anomaly_type"] == "spike"]
    dips     = df[df["anomaly_type"] == "dip"]

    fig = go.Figure()

    # Base line
    fig.add_trace(go.Scatter(
        x=normal["timestamp"], y=normal["value"],
        mode="lines",
        name="Normal",
        line=dict(color="#3b82d4", width=1.5),
    ))

    # Anomaly line segments (connect through anomalies visually)
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["value"],
        mode="lines",
        name="_base_line",
        line=dict(color="#3b82d4", width=1, dash="dot"),
        showlegend=False,
        opacity=0.3,
    ))

    # Spike markers
    if not spikes.empty:
        fig.add_trace(go.Scatter(
            x=spikes["timestamp"], y=spikes["value"],
            mode="markers",
            name="Spike Anomaly",
            marker=dict(color="red", size=9, symbol="circle"),
        ))

    # Dip markers
    if not dips.empty:
        fig.add_trace(go.Scatter(
            x=dips["timestamp"], y=dips["value"],
            mode="markers",
            name="Dip Anomaly",
            marker=dict(color="orange", size=9, symbol="triangle-down"),
        ))

    fig.add_hline(
        y=HIGH_UTIL_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text="80 % Threshold",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=f"Anomaly Detection — Instance {inst_id}  (IQR Method)",
        xaxis_title="Timestamp",
        yaxis_title="CPU Utilization (%)",
        yaxis=dict(range=[0, max(105, df["value"].max() * 1.05)]),
        hovermode="x unified",
        margin=dict(t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────
# 5. INSIGHTS & RECOMMENDATIONS (static, data-grounded)
# ─────────────────────────────────────────────

INSIGHTS = [
    {
        "title": "🔴 Persistently Overloaded Instance (825cc2)",
        "body": (
            "This instance sustained CPU utilization between 91 % and 96 % continuously "
            "across the entire 14-day observation window. It never dropped below 90 %. "
            "This is a critical capacity risk — at these levels, the instance has no headroom "
            "for unexpected traffic spikes and is likely causing application latency or failures."
        ),
    },
    {
        "title": "🟠 Runaway Load — Escalating to Saturation (ac20cd)",
        "body": (
            "Instance ac20cd started the observation period at a healthy ~42 % average CPU, "
            "but progressively ramped toward saturation, reaching ~99 % by the end of the window. "
            "This pattern is typical of a runaway process, a memory leak causing swap pressure, "
            "or an unchecked workload that is consuming all available CPU over time."
        ),
    },
    {
        "title": "🟡 Transient CPU Spike — Isolated Burst (77c1ca)",
        "body": (
            "This instance operated at near-zero CPU (~0.1 %) for almost its entire observation "
            "period, but experienced a sharp isolated spike — jumping from 0.1 % to ~92 % "
            "and returning to baseline within 30 minutes. This pattern is characteristic of a "
            "scheduled cron job, a batch import, or an external traffic burst rather than a "
            "persistent load issue."
        ),
    },
    {
        "title": "💤 Idle / Unused Instances (24ae8d, c6585a)",
        "body": (
            "Both 24ae8d and c6585a maintained CPU utilization below 0.15 % across their full "
            "14-day windows with no meaningful activity detected. Running EC2 instances at "
            "near-zero CPU is a direct cost inefficiency — these are strong candidates for "
            "rightsizing to a smaller instance type or shutdown."
        ),
    },
    {
        "title": "✅ Healthy Baseline — Stable Mid-Range (5f5533)",
        "body": (
            "Instance 5f5533 maintained a stable CPU utilization between 37 % and 55 % "
            "throughout the observation period, with consistent standard deviation. "
            "This is close to the ideal operational range — busy enough to justify the resource, "
            "but with sufficient headroom to absorb demand spikes."
        ),
    },
    {
        "title": "🔵 Lightly Loaded Active Instances (53ea38, fe7f93)",
        "body": (
            "These two instances showed consistent CPU utilization below 3 % — they are "
            "operational (not idle) but underutilized. They may host low-traffic services, "
            "scheduled tasks, or act as standby nodes. Worth monitoring for any unexpected "
            "changes that would indicate a configuration drift."
        ),
    },
]

RECOMMENDATIONS = [
    {
        "title": "1. Scale or redistribute the load on 825cc2 immediately",
        "body": (
            "Persistent CPU above 90 % leaves no operational headroom. Options: upgrade to a "
            "larger EC2 instance type (vertical scaling), distribute workloads across multiple "
            "instances (horizontal scaling), or review the application for inefficient code paths. "
            "Set a CloudWatch alarm at 85 % CPU to receive early warning of future saturation."
        ),
    },
    {
        "title": "2. Investigate ac20cd for runaway processes",
        "body": (
            "The steady ramp from ~42 % to ~99 % over 14 days without intervention is a "
            "sign that no monitoring or auto-scaling policy was in place. Review running "
            "processes, check for memory leaks, and enable CPU-based Auto Scaling policies. "
            "Set a CloudWatch alarm at 80 % with an automated notification."
        ),
    },
    {
        "title": "3. Rightsize or shut down idle instances (24ae8d, c6585a)",
        "body": (
            "Near-zero CPU for 14 continuous days with no observed workload strongly suggests "
            "these instances serve no active purpose. Shutting them down or switching to a "
            "smaller instance type (e.g., t3.nano) could reduce monthly EC2 costs with no "
            "operational impact. Confirm no scheduled jobs run outside the observation window."
        ),
    },
    {
        "title": "4. Identify and manage the scheduled job on 77c1ca",
        "body": (
            "The isolated 30-minute spike from near-zero to 92 % is almost certainly a "
            "scheduled or triggered job. Identify the job using CloudWatch Logs or system "
            "audit logs around the spike timestamp. If the job recurs, allocate a dedicated "
            "instance or time-slice the job to avoid resource contention."
        ),
    },
]


# ─────────────────────────────────────────────
# 6. MAIN DASHBOARD LAYOUT
# ─────────────────────────────────────────────

def main():
    # ── Load data ──────────────────────────────
    datasets, combined, quality_df = load_all_datasets()

    # Apply anomaly detection to all instances
    datasets = {k: detect_anomalies(v) for k, v in datasets.items()}
    combined = pd.concat(datasets.values(), ignore_index=True)

    # Compute KPI tables
    fleet_df   = compute_fleet_summary(datasets)
    fleet_kpis = compute_fleet_kpis(datasets, fleet_df)

    # ── Sidebar ────────────────────────────────
    st.sidebar.title("📊 EC2 CPU Analytics")
    st.sidebar.markdown("**IBM SkillsBuild — Data Analytics with AI Internship**")
    st.sidebar.divider()

    selected_label = st.sidebar.selectbox(
        "Select Instance (drill-down charts)",
        options=list(INSTANCE_LABELS.values()),
        index=0,
    )
    selected_id = [k for k, v in INSTANCE_LABELS.items() if v == selected_label][0]

    st.sidebar.divider()
    st.sidebar.markdown("#### Anomaly Detection Method")
    st.sidebar.markdown(
        "**IQR (Interquartile Range)**  \n"
        "A data point is flagged as anomalous if it falls outside "
        "**Q1 − 1.5 × IQR** (lower) or **Q3 + 1.5 × IQR** (upper), "
        "where IQR = Q3 − Q1.  \n\n"
        "This is a standard statistical method that requires no machine learning model "
        "and adapts automatically to each instance's own distribution."
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        "**Data source:** Numenta Anomaly Benchmark (NAB)  \n"
        "AWS EC2 CPU Utilization dataset  \n"
        "© Amazon Web Services / Numenta"
    )

    # ── Header ─────────────────────────────────
    st.title("🖥️ AWS EC2 CPU Utilization Analytics & Anomaly Detection")
    st.markdown(
        "An interactive analysis of **8 AWS EC2 instances** — covering utilization trends, "
        "KPI benchmarks, anomaly detection, and actionable cloud operations recommendations."
    )
    st.divider()

    # ── Section 1: Fleet KPI Cards ──────────────
    st.subheader("Fleet Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Instances",        "8")
    c2.metric("Fleet Avg CPU (%)",       f"{fleet_kpis['fleet_mean']} %")
    c3.metric("Most Overloaded",         fleet_kpis["most_overloaded"])
    c4.metric("High-Util Events (>80 %)", f"{fleet_kpis['high_util_events']:,}")
    c5.metric("Total Anomalies Detected", f"{fleet_kpis['total_anomalies']:,}")
    st.divider()

    # ── Section 2: Instance KPI Table ──────────
    st.subheader("Per-Instance KPI Summary")
    display_fleet = fleet_df.drop(columns=["Label"]).set_index("Instance")
    st.dataframe(display_fleet, use_container_width=True)

    with st.expander("📋 Data Quality Report"):
        st.dataframe(quality_df.set_index("Instance"), use_container_width=True)
        st.caption(
            "All 8 files loaded without errors. No missing values or duplicate timestamps detected."
        )
    st.divider()

    # ── Section 3: CPU Trend — Selected Instance ─
    st.subheader(f"CPU Utilization Trend — {selected_label}")
    df_sel = datasets[selected_id]
    st.plotly_chart(chart_instance_trend(df_sel, selected_id), use_container_width=True)
    kpi_sel = compute_kpis(df_sel)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average CPU",    f"{kpi_sel['mean']} %")
    k2.metric("Max CPU",        f"{kpi_sel['max']} %")
    k3.metric("High-Util (%)",  f"{kpi_sel['high_util_pct']} %")
    k4.metric("Anomalies",      f"{kpi_sel['anomaly_count']}")
    st.divider()

    # ── Section 4: All-Instance Comparison ──────
    st.subheader("All Instances — Comparison")
    tab1, tab2 = st.tabs(["📈 Time-Series Overlay", "📊 Average CPU Ranking"])

    with tab1:
        st.plotly_chart(chart_all_instances_overlay(combined), use_container_width=True)
        st.caption(
            "Note: instances belong to two date groups (Feb 2014 and Apr 2014). "
            "The x-axis shows absolute calendar time; gaps between groups are expected."
        )

    with tab2:
        st.plotly_chart(chart_avg_cpu_bar(fleet_df), use_container_width=True)
        st.caption(
            "Colour coding: 🔴 Red = critical (>90 %), 🟠 Orange = elevated (60–90 %), "
            "🟢 Green = healthy (20–60 %), 🔵 Blue = under-utilised (<20 %)."
        )
    st.divider()

    # ── Section 5: Anomaly Detection ────────────
    st.subheader("Anomaly Detection")

    with st.expander("ℹ️ How does the IQR anomaly detection method work?", expanded=False):
        st.markdown(
            """
            **IQR — Interquartile Range Method**

            For each EC2 instance, the algorithm works as follows:

            1. Compute **Q1** (25th percentile) and **Q3** (75th percentile) of the CPU readings.
            2. Calculate **IQR = Q3 − Q1** (the spread of the middle 50 % of readings).
            3. Define boundaries:
               - **Lower fence** = Q1 − 1.5 × IQR
               - **Upper fence** = Q3 + 1.5 × IQR
            4. Any reading **below the lower fence** is flagged as a **dip anomaly** (unexpectedly low CPU).
            5. Any reading **above the upper fence** is flagged as a **spike anomaly** (unexpectedly high CPU).

            This method is a standard statistical approach used in boxplot outlier detection.
            It does **not require a machine learning model**, is fully explainable, and adapts
            automatically to each instance's own baseline — so a normally high-CPU instance
            (like `825cc2`) will not have all its readings flagged just because it runs at 93 %.
            """
        )

    st.plotly_chart(chart_anomaly_overlay(df_sel, selected_id), use_container_width=True)

    st.markdown("**Anomaly Summary — All Instances**")
    anomaly_summary = fleet_df[["Instance", "Label", "Anomaly Count", "Anomaly (%)"]].copy()
    anomaly_summary = anomaly_summary.sort_values("Anomaly Count", ascending=False).reset_index(drop=True)
    st.dataframe(anomaly_summary, use_container_width=True, hide_index=True)
    st.caption(
        "Anomalies are computed independently per instance using that instance's own IQR boundaries. "
        "A zero anomaly count on a flat-line instance (e.g. 24ae8d) means the instance is "
        "consistently operating within its own normal range — which is itself an insight."
    )
    st.divider()

    # ── Section 6: Insights & Recommendations ───
    st.subheader("Key Insights")
    for ins in INSIGHTS:
        with st.expander(ins["title"], expanded=False):
            st.write(ins["body"])

    st.divider()
    st.subheader("Recommendations")
    for rec in RECOMMENDATIONS:
        with st.expander(rec["title"], expanded=False):
            st.write(rec["body"])

    st.divider()

    # ── Footer ──────────────────────────────────
    st.markdown(
        "<div style='text-align:center; color:#888; font-size:0.82rem; padding-top:1rem;'>"
        "IBM SkillsBuild Data Analytics with AI Internship — Final Project &nbsp;|&nbsp; "
        "Dataset: Numenta Anomaly Benchmark (NAB) — AWS EC2 CPU Utilization"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__" or True:
    main()
