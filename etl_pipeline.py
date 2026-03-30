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

def load_master_data(data_dir: str = "new/Cleaned", load_all: bool = True) -> pd.DataFrame:
    """
    Load all CSV files from the Cleaned folder and create a single master dataframe.
    """
    file_pattern = os.path.join(data_dir, "*_frac.csv")
    csv_files = sorted(glob.glob(file_pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir} matching pattern {file_pattern}")
    
    print(f"Found {len(csv_files)} CSV files to load...")
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, engine='c')
            print(f"  Loaded: {os.path.basename(f)} ({len(df)} rows)")
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {f}: {e}")
    
    if not dfs:
        raise ValueError("No CSV files could be loaded successfully")
    
    master_df = pd.concat(dfs, ignore_index=True)
    return master_df

def load_feature_table(
    data_dir: str = "new/Cleaned",
    sample_frac: float = 1.0,
    chunk_size: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load the Cleaned dataset in a memory-safe manner.
    """
    file_pattern = os.path.join(data_dir, "*_frac.csv")
    csv_files = sorted(glob.glob(file_pattern))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith(".~lock")]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching: {file_pattern}")

    rng = np.random.default_rng(seed)
    sampled_chunks: list[pd.DataFrame] = []

    for filepath in csv_files:
        fname = os.path.basename(filepath)
        print(f"  → Reading {fname} …", flush=True)

        try:
            reader = pd.read_csv(
                filepath,
                usecols=lambda c: c in _NEEDED_COLS,
                dtype=_DTYPES,
                chunksize=chunk_size,
                on_bad_lines="skip",
            )
            college_name = os.path.basename(filepath).replace("Student_Book_Interactions_", "").replace("_frac.csv", "").upper()
            for chunk in reader:
                n_keep = max(1, int(len(chunk) * sample_frac))
                sampled = chunk.sample(n=n_keep, random_state=int(rng.integers(0, 2**31)))
                sampled["College"] = college_name
                sampled_chunks.append(sampled)
                del chunk
                gc.collect()

        except Exception as exc:
            print(f"    ⚠ Skipping {fname}: {exc}")

    if not sampled_chunks:
        return pd.DataFrame()

    df = pd.concat(sampled_chunks, ignore_index=True)
    del sampled_chunks
    gc.collect()

    print(f"  ✓ Sampled {len(df):,} rows from {len(csv_files)} files. Engineering features …")

    # --- Categorical mappings ---
    df["Term"]     = (df["term_code"].astype(str) + " " + df["term_year"].astype(str)).astype("category")
    df["Year"]     = df["term_year"].astype(str).astype("category")
    df["Semester"] = df["term_code"].map({"F": "Fall", "W": "Winter", "S": "Spring", "A": "Annual"}).fillna("Other").astype("category")

    def _dept(sec: str) -> str:
        parts = str(sec).split("-")
        return parts[3] if len(parts) > 3 else "GEN"

    df["Dept_Code"]    = df["section_id"].map(_dept).astype("category")
    df["College"]      = df["College"].astype("category")
    df["Title"]        = df["title"].fillna("Unknown Title").astype("category")
    df["Publisher"]    = df["author"].fillna("Unknown").astype(str).str.slice(0, 20).astype("category")
    df["Student_Type"] = df["student_full_part_time_status"].map(
        {"F": "Full-Time", "P": "Part-Time", "H": "Half-Time"}
    ).fillna("Full-Time")
    df["Format"] = df["ebook_ind"].fillna(0).map(lambda x: "Digital" if x == 1.0 else "Physical")

    # --- Economic features ---
    rn  = df["retail_new"].fillna(100.0).clip(lower=0.01)
    rr  = df["retail_new_rent"].fillna(50.0).clip(lower=0.0)

    df["Rental_to_Retail_Ratio"] = (rr / rn).clip(0.0, 1.5).astype("float32")
    df["Arbitrage_Index"]        = (1.0 - df["Rental_to_Retail_Ratio"]).astype("float32")

    aff = df["price_affordability_score"].fillna(300.0)
    df["Wallet_Pressure_Score"]  = (aff / 1000.0).clip(0.0, 1.0).astype("float32")
    df["Digital_Lock_Flag"]      = df["ebook_ind"].fillna(0.0).astype("float32")

    # --- Labels ---
    wb = df["will_buy"].fillna(1.0)
    df["Actual_Purchase_Flag"]   = wb.astype("float32")
    df["Opt_Out_Probability"]    = (1.0 - wb).astype("float32")

    # --- Demand / spend placeholders ---
    df["Predicted_Demand_Units"] = 1
    df["Unit_Price"]             = rn.astype("float32")
    df["Predicted_Purchase_Prob"] = wb.astype("float32")
    df["Projected_Spend"]        = (df["Unit_Price"] * wb).astype("float32")

    # --- Raw ML inputs ---
    df["family_annual_income"] = df["family_annual_income"].fillna(40_000.0).astype("float32")
    df["has_scholarship"]      = df["has_scholarship"].fillna(0.0).astype("float32")
    df["has_loan"]             = df["has_loan"].fillna(0.0).astype("float32")
    df["is_rental"]            = df["is_rental"].fillna(0.0).astype("float32")

    # --- Synthetic proxies ---
    df["Major_Alignment_Score"] = rng.uniform(0.5, 1.0, size=len(df)).astype("float32")
    df["Commuter_Friction"]     = rng.uniform(0.1, 0.9, size=len(df)).astype("float32")

    output_cols = [
        "Term", "Year", "Semester", "College", "Dept_Code", "Title", "Publisher", "Student_Type", "Format",
        "Rental_to_Retail_Ratio", "Arbitrage_Index", "Wallet_Pressure_Score",
        "Digital_Lock_Flag", "Actual_Purchase_Flag", "Opt_Out_Probability",
        "Predicted_Demand_Units", "Unit_Price", "Predicted_Purchase_Prob",
        "Projected_Spend", "family_annual_income", "has_scholarship",
        "has_loan", "is_rental", "Major_Alignment_Score", "Commuter_Friction",
    ]

    return df[output_cols].reset_index(drop=True)

__all__ = ["load_master_data", "load_feature_table"]
