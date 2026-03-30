# Predictive Procurement Analytics Dashboard

This project implements a **Streamlit + Plotly** dashboard for your _University Bulk Order & Predictive Procurement Analytics System_. It is designed for an M.Tech AI/ML portfolio and demonstrates the full analytics workflow: data ingestion, feature engineering, model scoring, and decision support for textbook procurement.

## Features

- Executive KPIs (predicted demand, projected spend, digital vs physical mix, high‑risk opt‑out rate).
- Interactive filters by term, department, publisher, student type, etc.
- Price sensitivity & opt‑out scatter plot (Arbitrage Index vs Opt‑Out Probability).
- Feature importance view for the ML model (Random Forest / XGBoost).
- Format preference charts (digital vs physical breakdown by segment).
- Risk segmentation by department and course.

## Project Structure

- `schema.sql` – Snowflake‑style schema (fact ENROLLMENTS + dimension tables).
- `etl_pipeline.py` – Placeholder for data cleaning, joins, and feature table creation.
- `feature_engine.py` – Placeholder for feature engineering (Arbitrage Index, Wallet Pressure, etc.).
- `dashboard_app.py` – Main Streamlit app with Plotly visualizations.

## Getting Started

1. **Create a virtual environment (recommended)**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**  
   ```bash
   streamlit run dashboard_app.py
   ```

## Plugging in Your Real Data

- Replace the dummy data generation in `dashboard_app.py` with a call to your ETL/feature code, for example:
  - `from etl_pipeline import load_enrollment_fact_table`
  - `from feature_engine import build_feature_table`
- Ensure the final DataFrame used in the app has, at minimum, the following columns (you can add more as needed):
  - `Term`, `Dept_Code`, `Publisher`, `Student_Type`, `Format` (Digital/Physical)
  - `Predicted_Demand`, `Projected_Spend`, `Opt_Out_Probability`
  - `Arbitrage_Index`, `Wallet_Pressure_Score`, `Digital_Lock_Flag`, `Major_Alignment_Score`, `Commuter_Friction`


