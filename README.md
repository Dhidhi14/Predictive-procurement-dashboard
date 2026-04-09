# 📦 University Bulk Order & Predictive Procurement Analytics System

A data-driven Streamlit dashboard designed to help universities analyze bulk purchasing patterns, forecast procurement needs, and gain actionable insights from historical order and sentiment data.

---

<<<<<<< HEAD
## 🚀 Features
=======
- Executive KPIs (predicted demand, projected spend, digital vs physical mix, high‑risk opt‑out rate).
- Interactive filters by term, department, publisher, student type, safely handled explicitly via real-time data cleansing.
- Price sensitivity & opt‑out scatter plot (Arbitrage Index vs Opt‑Out Probability).
- Feature importance view for the ML model (Random Forest).
- Format preference charts (digital vs physical volume breakdown).
- Risk segmentation by department factored directly against real predicted units.
- Absolute True Weighted Average ROI calculations to eliminate multi-tier price biases.
>>>>>>> feat: stabilize hybrid data pipeline, refine UI widgets, and enforce deep mathematical consistency across Streamlit dashboard

- **Predictive Procurement Analytics** — Forecasts future bulk order demand using historical purchase data
- **KPI Dashboard** — Pre-computed key performance indicators displayed across interactive charts and metrics
- **Sentiment Enrichment** — Integrates book/resource sentiment scores to inform procurement decisions
- **Student Purchase Tracking** — Monitors and counts student-level purchasing behavior
- **ETL Pipeline** — Automated data ingestion, transformation, and loading from raw master data
- **Feature Engineering** — Derived features for model training and analytics
- **Model Evaluation** — Built-in tools to assess predictive model performance
- **Large Dataset Support** — Chunked processing handles 7.5M+ records efficiently

---

## 🗂️ Project Structure

```bash
Predictive-procurement-dashboard/
│
├── dashboard_app.py            # Main Streamlit dashboard application
├── etl_pipeline.py             # ETL: data ingestion and transformation
├── feature_engine.py           # Feature engineering for ML models
├── precompute_kpis.py          # Pre-computes KPIs from master data
├── enrich_sentiment.py         # Aggregates and enriches sentiment scores
├── model_evaluation.py         # Model performance evaluation utilities
├── count_student_purchase.py   # Student purchase frequency analysis
├── schema.sql                  # Database schema definition
├── requirements.txt            # Python dependencies
│
├── new/
│   └── master_data/
│       ├── master_data.csv             # Primary raw procurement dataset
│       └── Training_Data_Clean.csv     # Cleaned sentiment training data
│
├── resource/                   # Auto-generated output files
│   ├── summary_kpis.csv        # Pre-computed KPI summary
│   └── book_sentiment.csv      # Aggregated sentiment scores
│
└── my_env/                     # Python virtual environment
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.12+
- PowerShell (Windows) or Terminal (macOS/Linux)

### 1. Clone the Repository

```bash
git clone https://github.com/Ashishparmar265/Predictive-procurement-dashboard.git
cd Predictive-procurement-dashboard
```

### 2. Create a Virtual Environment

```bash
python -m venv my_env
```

### 3. Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
.\my_env\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
my_env\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source my_env/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

> Make sure your virtual environment is activated before running any script.

### Step 1 — Pre-compute KPIs

Processes the master dataset in chunks and generates `resource/summary_kpis.csv`:

```bash
python precompute_kpis.py
```

> Processes ~7.5 million records. Expect a few minutes depending on your machine.

### Step 2 — Enrich Sentiment Data

Aggregates sentiment scores and saves them to `resource/book_sentiment.csv`:

```bash
python enrich_sentiment.py
```

### Step 3 — Launch the Dashboard

```bash
streamlit run dashboard_app.py
```

Then open your browser and navigate to:

```bash
Local:    http://localhost:8501
Network:  http://<your-ip>:8501
```

---

## 📋 Data Requirements

Place the following files before running any scripts:

| File | Location |
|------|----------|
| `master_data.csv` | `new/master_data/master_data.csv` |
| `Training_Data_Clean.csv` | `new/master_data/Training_Data_Clean.csv` |

> ⚠️ **Windows path tip:** Always use forward slashes (`/`) or raw strings (`r"..."`) in Python scripts to avoid invalid escape sequence errors with backslash paths.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Streamlit | Interactive web dashboard |
| Pandas | Data manipulation |
| Dask | Large-scale chunked data processing |
| Plotly | Data visualization |
| scikit-learn | Predictive modeling |
| WordCloud | Sentiment visualization |

---

## 📌 Notes

- The `resource/` folder is auto-generated. Do not manually edit files inside it.
- Always run `precompute_kpis.py` and `enrich_sentiment.py` before launching the dashboard for the first time, or whenever the source data changes.
- If you encounter a `SyntaxWarning: invalid escape sequence` error, ensure all file paths in `.py` files use forward slashes or raw strings.