"""
ETL pipeline for the Predictive Procurement Analytics project.

Memory-safe strategy:
- Reads each of the 11 Cleaned CSVs ONE AT A TIME with chunked iteration.
- Samples a small fraction (default 1%) from each chunk before appending.
- Never holds more than one chunk in RAM at a time.
- Returns a single small Pandas DataFrame suitable for ML and the dashboard.
"""

from __future__ import annotations

import gc
import glob
import os

import numpy as np
import pandas as pd

# Columns we actually need — reading only these avoids loading unnecessary data
_NEEDED_COLS = [
    "sis_user_id",
    "section_id",
    "term_code",
    "term_year",
    "title",
    "author",
    "ebook_ind",
    "retail_new",
    "retail_new_rent",
    "price_affordability_score",
    "family_annual_income",
    "has_scholarship",
    "has_loan",
    "is_rental",
    "will_buy",
    "student_full_part_time_status",
]

_DTYPES = {
    "sis_user_id":               "category",
    "section_id":                "category",
    "term_code":                 "category",
    "term_year":                 "category",
    "author":                    "category",
    "student_full_part_time_status": "category",
    "College":                   "category",
    "Year":                      "category",
    "Semester":                  "category",
    "Department":                "category",
    "ebook_ind":                 "float32",
    "retail_new":                "float32",
    "retail_new_rent":           "float32",
    "price_affordability_score": "float32",
    "family_annual_income":      "float32",
    "has_scholarship":           "float32",
    "has_loan":                  "float32",
    "is_rental":                 "float32",
    "will_buy":                  "float32",
}

def load_summary_kpis(summary_path: str = "resource/summary_kpis.csv") -> pd.DataFrame:
    """Load the pre-computed KPI summary for accurate global totals."""
    if os.path.exists(summary_path):
        return pd.read_csv(summary_path)
    return pd.DataFrame()

def load_feature_table(
    data_path: str = "new/master_data/master_data_sampled.csv",
    sample_limit: int = 20_000,
) -> pd.DataFrame:
    """
    Load a sampled subset of the master data for ML modeling and simulation.
    This reads from a pre-sampled 60k row subset generated securely via bash `shuf`.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Optimized sample not found at {data_path}. Please run the sampler first.")

    print(f"  → Loading optimized ML sample ({data_path}) …", flush=True)

    # Use explicit list for usecols to be faster and safer
    target_cols = [c for c in _NEEDED_COLS if c != "College" and c != "Year" and c != "Semester" and c != "Department"]
    # Add metadata cols back if they are not in _NEEDED_COLS
    for meta in ["College", "Year", "Semester", "Department"]:
        if meta not in target_cols:
            target_cols.append(meta)

    # Read the pre-sampled file
    df = pd.read_csv(
        data_path,
        usecols=lambda c: c in target_cols or c == "will_buy",
        dtype=_DTYPES,
        engine='c' # Use fast C engine
    )
    
    # Sample down to the strict memory limit
    if len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=42)
    
    gc.collect() # Immediate cleanup
    
    # Remove year '21' or 2021 data
    if "Year" in df.columns:
        df["Year"] = df["Year"].astype(str)
        df = df[~df["Year"].isin(["21", "2021", "21.0"])]
    if "term_year" in df.columns:
        df["term_year"] = df["term_year"].astype(str)
        df = df[~df["term_year"].isin(["21", "2021", "21.0"])]
        
    df = df.copy() # Ensure memory contiguous


    # --- Data Cleaning & Feature Generation (Matching precompute logic) ---
    df["Term"]         = (df["term_code"].astype(str) + " " + df["term_year"].astype(str)).astype("category")
    df["Title"]        = df["title"].fillna("Unknown Title").astype("category")
    df["Publisher"]    = df["author"].astype(str).fillna("Unknown").str.slice(0, 20).astype("category")
    df["Dept_Code"]    = df["Department"].astype("category")
    df["Student_Type"] = df["student_full_part_time_status"].map(
        {"F": "Full-Time", "P": "Part-Time", "H": "Half-Time"}
    ).fillna("Full-Time")
    df["Format"] = df["ebook_ind"].fillna(0).map(lambda x: "Digital" if x == 1.0 else "Physical")

    # Economic metrics
    rn = df["retail_new"].fillna(100.0).clip(lower=0.01)
    df["Unit_Price"] = rn.astype("float32")
    df["Projected_Spend"] = (df["Unit_Price"] * df["will_buy"].fillna(1)).astype("float32")
    df["Opt_Out_Probability"] = (1.0 - df["will_buy"].fillna(1)).astype("float32")
    df["Actual_Purchase_Flag"] = df["will_buy"].fillna(1).astype("float32")
    df["Predicted_Demand_Units"] = 1
    df["Predicted_Purchase_Prob"] = df["will_buy"].fillna(1).astype("float32")

    # Inherit existing logic for simulated fields
    df["Rental_to_Retail_Ratio"] = (df["retail_new_rent"].fillna(50) / rn).clip(0.0, 1.5).astype("float32")
    df["Arbitrage_Index"] = (1.0 - df["Rental_to_Retail_Ratio"]).astype("float32")
    df["Wallet_Pressure_Score"] = (df["price_affordability_score"].fillna(300) / 1000.0).clip(0.0, 1.0).astype("float32")
    df["Digital_Lock_Flag"] = df["ebook_ind"].fillna(0.0).astype("float32")
    
    # ML specific inputs
    df["family_annual_income"] = df["family_annual_income"].fillna(40_000.0).astype("float32")
    df["has_scholarship"] = df["has_scholarship"].fillna(0.0).astype("float32")
    df["has_loan"] = df["has_loan"].fillna(0.0).astype("float32")
    df["is_rental"] = df["is_rental"].fillna(0.0).astype("float32")

    # --- Sentiment Enrichment Join ---
    sentiment_path = "resource/book_sentiment.csv"
    if os.path.exists(sentiment_path):
        sdf = pd.read_csv(sentiment_path)
        # Prepare for join by standardizing keys
        df["title_upper"] = df["title"].astype(str).str.upper().str.strip()
        sdf["Book_Title"] = sdf["Book_Title"].astype(str).str.upper().str.strip()
        
        df = df.merge(sdf, left_on="title_upper", right_on="Book_Title", how="left")
        
        # Fill missing sentiment with neutral values (3.0 out of 5.0)
        sent_cols = [c for c in sdf.columns if c != "Book_Title"]
        for scol in sent_cols:
            df[scol] = df[scol].fillna(3.0).astype("float32")
            
        df.drop(columns=["title_upper", "Book_Title"], inplace=True)
    else:
        sent_cols = ["Book_Rating", "Sentiment_Understandability", "Sentiment_Value_For_Money", "Sentiment_Exam_Utility", "Sentiment_Avg_Rating"]
        for scol in sent_cols:
            df[scol] = 3.0

    output_cols = [
        "Term", "Year", "Semester", "College", "Dept_Code", "Title", "Publisher", "Student_Type", "Format",
        "Rental_to_Retail_Ratio", "Arbitrage_Index", "Wallet_Pressure_Score",
        "Digital_Lock_Flag", "Actual_Purchase_Flag", "Opt_Out_Probability",
        "Predicted_Demand_Units", "Unit_Price", "Predicted_Purchase_Prob",
        "Projected_Spend", "family_annual_income", "has_scholarship",
        "has_loan", "is_rental"
    ] + sent_cols

    return df[output_cols].reset_index(drop=True)

__all__ = ["load_master_data", "load_feature_table", "load_summary_kpis"]
