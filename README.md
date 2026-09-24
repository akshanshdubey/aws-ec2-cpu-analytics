# AWS EC2 CPU Utilization Analytics & Anomaly Detection

**IBM SkillsBuild Data Analytics with AI Internship — Final Project**

---

## Project Overview

This project analyses CPU utilization data from **8 AWS EC2 instances** sourced from the
[Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) dataset.

The application:
- Loads and cleans all 8 CSV time-series files
- Computes KPI statistics per instance and across the fleet
- Visualises CPU trends and compares all instances side-by-side
- Detects anomalies using the IQR (Interquartile Range) statistical method
- Surfaces actionable cloud-operations insights and recommendations
- Presents everything in a clean, interactive **Streamlit** web dashboard

---

## Dataset

| File | Instance ID | Date Range | Avg CPU Profile |
|---|---|---|---|
| ec2_cpu_utilization_5f5533.csv | 5f5533 | Feb 14–28, 2014 | ~43 % — stable medium load |
| ec2_cpu_utilization_24ae8d.csv | 24ae8d | Feb 14–28, 2014 | ~0.13 % — near-idle |
| ec2_cpu_utilization_53ea38.csv | 53ea38 | Feb 14–28, 2014 | ~1.8 % — very low use |
| ec2_cpu_utilization_77c1ca.csv | 77c1ca | Apr 2–16, 2014 | ~0.1 % with isolated spike to 92 % |
| ec2_cpu_utilization_825cc2.csv | 825cc2 | Apr 10–24, 2014 | ~90 % — persistently overloaded |
| ec2_cpu_utilization_ac20cd.csv | ac20cd | Apr 2–16, 2014 | ~41 % escalating to 99 % |
| ec2_cpu_utilization_c6585a.csv | c6585a | Apr 2–16, 2014 | ~0.09 % — near-zero |
| ec2_cpu_utilization_fe7f93.csv | fe7f93 | Feb 14–28, 2014 | ~5.8 % — low stable use |

- **4,032 rows per file** (5-minute intervals, ~14 days each)
- **Columns:** `timestamp`, `value` (CPU utilization %)
- **No missing values** across all 8 files

---

## Project Structure

```
project-root/
├── dataset/                         # Original CSV files (read-only)
│   ├── ec2_cpu_utilization_5f5533.csv
│   ├── ec2_cpu_utilization_24ae8d.csv
│   ├── ec2_cpu_utilization_53ea38.csv
│   ├── ec2_cpu_utilization_77c1ca.csv
│   ├── ec2_cpu_utilization_825cc2.csv
│   ├── ec2_cpu_utilization_ac20cd.csv
│   ├── ec2_cpu_utilization_c6585a.csv
│   └── ec2_cpu_utilization_fe7f93.csv
├── app.py                           # Single-file Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── project_report.docx              # Full project report
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

> **Note:** The `dataset/` folder must be in the same directory as `app.py`.

---

## Dashboard Sections

| Section | Description |
|---|---|
| **Fleet Overview** | 5 high-level KPI metric cards across all 8 instances |
| **Per-Instance KPI Table** | Mean, Min, Max, Std Dev, High-Util %, Anomaly count per instance |
| **CPU Trend** | Interactive line chart for the selected instance with 80 % threshold |
| **All Instances Comparison** | Multi-line overlay chart + Average CPU bar chart |
| **Anomaly Detection** | Anomaly chart for selected instance + summary table for all instances |
| **Insights** | 6 data-driven findings about the fleet |
| **Recommendations** | 4 actionable cloud operations recommendations |

---

## Anomaly Detection Method — IQR

The project uses the **Interquartile Range (IQR)** method to detect anomalies:

1. Compute **Q1** (25th percentile) and **Q3** (75th percentile) for each instance.
2. Calculate **IQR = Q3 − Q1**.
3. Flag any reading as anomalous if:
   - `value > Q3 + 1.5 × IQR` → **spike** anomaly
   - `value < Q1 − 1.5 × IQR` → **dip** anomaly

This method requires no machine learning model, is fully interpretable, and automatically
adapts to each instance's own distribution.

---

## Key Findings

- **Instance 825cc2** had sustained very high CPU utilization, with an average of ~90% and peaks above 99%, indicating a significant workload/overload risk.
- **Instance ac20cd** increased from roughly 42% to nearly 100% CPU, indicating a period of rapidly increasing workload that warrants investigation.
- **Instance 77c1ca** had one brief but extreme spike (0.1 % → 92 % → back to 0.1 %) in 30 minutes.
- **Instances 24ae8d and c6585a** were near-idle throughout — potential candidates for rightsizing or shutdown.

---

## Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.12+ | Core language |
| Streamlit | ≥1.32 | Web dashboard framework |
| Pandas | ≥2.0 | Data loading and manipulation |
| NumPy | ≥1.26 | Numerical computation |
| Plotly Express | ≥5.18 | Interactive charts |

---

## Data Source

Numenta Anomaly Benchmark (NAB) — AWS EC2 CPU Utilization dataset.
Original data collected via AWS CloudWatch.
Repository: https://github.com/numenta/NAB
