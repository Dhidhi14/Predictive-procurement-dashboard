"""
ETL pipeline for the Predictive Procurement Analytics project.

This module ingests real augmented data from the university SIS/ERP,
cleans/normalizes the datasets, and prepares feature tables for the ML models
and dashboard.
"""

from __future__ import annotations

import glob
import os
import pandas as pd
import numpy as np

def load_feature_table(data_dir: str = "new/Augmented", sample_frac: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """
    Load and map the augmented real dataset to the dashboard schema.
    Since the dataset is very large, sample_frac allows loading a subset to keep the dashboard responsive.
    """
    file_pattern = os.path.join(data_dir, "*_balanced_dataset.csv")
    csv_files = glob.glob(file_pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching {file_pattern}")
        
    dfs = []
    # Use a fixed random generator
    rng = np.random.default_rng(seed)
    
    for f in csv_files:
        try:
            # We sample a fraction to keep memory footprint low
            df = pd.read_csv(f, engine='c')
            if sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=seed)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Map physical columns to what the dashboard and model expect
    mapped_df = pd.DataFrame()
    
    mapped_df["Term"] = df["term_code"].fillna("Unknown") + " " + df["term_year"].astype(str)
    
    # Extract dept from section_id if possible (e.g. BN-8304-1-126-A-... -> 126 is dept?)
    def extract_dept(sec):
        parts = str(sec).split("-")
        return parts[3] if len(parts) > 3 else "GEN"
    mapped_df["Dept_Code"] = df["section_id"].apply(extract_dept)
    
    mapped_df["Publisher"] = df["author"].fillna("Unknown Author").astype(str).str.slice(0, 20)
    
    student_type_map = {"F": "Full-Time", "P": "Part-Time", "H": "Half-Time", "L": "Unknown"}
    mapped_df["Student_Type"] = df["student_full_part_time_status"].map(student_type_map).fillna("Full-Time")
    
    mapped_df["Format"] = df["ebook_ind"].apply(lambda x: "Digital" if x == 1.0 else "Physical")
    
    # Pricing & Economic Features
    retail_new = pd.to_numeric(df["retail_new"], errors='coerce').fillna(100.0)
    retail_rent = pd.to_numeric(df["retail_new_rent"], errors='coerce').fillna(50.0)
    
    retail_new_safe = retail_new.replace(0.0, 100.0)
    ratio = retail_rent / retail_new_safe
    mapped_df["Rental_to_Retail_Ratio"] = ratio.clip(0.0, 1.5)
    
    # Avoid zero division Arbitrage_Index
    mapped_df["Arbitrage_Index"] = 1.0 - mapped_df["Rental_to_Retail_Ratio"]
    
    # Wallet pressure
    afford_score = pd.to_numeric(df["price_affordability_score"], errors='coerce').fillna(300.0)
    max_score = afford_score.max() if afford_score.max() > 0 else 1.0
    mapped_df["Wallet_Pressure_Score"] = (afford_score / max_score).clip(0.0, 1.0)
    
    mapped_df["Digital_Lock_Flag"] = df["ebook_ind"].fillna(0.0)
    
    # Synthetic proxies
    mapped_df["Major_Alignment_Score"] = rng.uniform(0.5, 1.0, size=len(mapped_df))
    mapped_df["Commuter_Friction"] = rng.uniform(0.1, 0.9, size=len(mapped_df))
    
    # Labels
    will_buy = pd.to_numeric(df["will_buy"], errors='coerce').fillna(1)
    mapped_df["Actual_Purchase_Flag"] = will_buy
    mapped_df["Opt_Out_Probability"] = 1.0 - will_buy
    
    mapped_df["Predicted_Demand_Units"] = 1
    mapped_df["Unit_Price"] = retail_new.clip(0.01) # Avoid zero price entirely
    
    # By default set pred to actual, ML model replaces this
    mapped_df["Predicted_Purchase_Prob"] = will_buy
    mapped_df["Projected_Spend"] = mapped_df["Predicted_Demand_Units"] * mapped_df["Unit_Price"] * mapped_df["Predicted_Purchase_Prob"]
    
    # Raw features for ML
    mapped_df["family_annual_income"] = pd.to_numeric(df["family_annual_income"], errors='coerce').fillna(40000)
    mapped_df["has_scholarship"] = pd.to_numeric(df["has_scholarship"], errors='coerce').fillna(0)
    mapped_df["has_loan"] = pd.to_numeric(df["has_loan"], errors='coerce').fillna(0)
    mapped_df["is_rental"] = pd.to_numeric(df["is_rental"], errors='coerce').fillna(0)
    
    return mapped_df

__all__ = ["load_feature_table"]
