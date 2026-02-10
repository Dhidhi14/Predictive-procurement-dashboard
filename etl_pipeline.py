"""
ETL pipeline placeholder for the Predictive Procurement Analytics project.

In your full project, this module should:
1. Ingest raw data from the university SIS/ERP and bookstore systems.
2. Clean and normalize datasets (students, sections, adoptions, enrollment transactions).
3. Join them into a fact table at the grain: one row per student per section.
4. Persist the resulting tables into the schema defined in `schema.sql`.

For the dashboard demo, we expose a simple `load_feature_table` function that you
can later replace with your real ETL / feature-store read logic.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def load_feature_table(seed: int = 42, n_rows: int = 2000) -> pd.DataFrame:
    """
    Generate a synthetic feature table that mimics the structure of the
    real predictive procurement dataset.

    Replace this with a real database query or file read in production.
    """

    rng = np.random.default_rng(seed)

    terms = ["Fall 2025", "Spring 2026", "Fall 2026"]
    departments = ["CSE", "HUM", "ART", "MUS", "ENG", "BIO", "BUS"]
    publishers = ["Pearson", "McGraw-Hill", "Cengage", "Wiley"]
    student_types = ["Full-Time", "Part-Time"]
    formats = ["Physical", "Digital"]

    df = pd.DataFrame(
        {
            "Term": rng.choice(terms, size=n_rows),
            "Dept_Code": rng.choice(departments, size=n_rows),
            "Publisher": rng.choice(publishers, size=n_rows),
            "Student_Type": rng.choice(student_types, size=n_rows),
            "Format": rng.choice(formats, size=n_rows, p=[0.7, 0.3]),
        }
    )

    # Economic features
    df["Rental_to_Retail_Ratio"] = rng.uniform(0.4, 1.1, size=n_rows)
    df["Wallet_Pressure_Score"] = rng.uniform(0, 1, size=n_rows)

    # Behavioral / logistical features
    df["Digital_Lock_Flag"] = rng.integers(0, 2, size=n_rows)
    df["Major_Alignment_Score"] = rng.uniform(0, 1, size=n_rows)
    df["Commuter_Friction"] = rng.uniform(0, 1, size=n_rows)

    # Arbitrage Index: lower ratio => more arbitrage opportunity
    df["Arbitrage_Index"] = 1 - df["Rental_to_Retail_Ratio"]

    # Construct an opt-out probability that depends on the features
    base = 0.25 + 0.4 * df["Arbitrage_Index"] + 0.3 * df["Wallet_Pressure_Score"]
    base -= 0.35 * df["Digital_Lock_Flag"]  # digital lock => more likely to opt-in
    base -= 0.15 * df["Major_Alignment_Score"]
    base = base.clip(0.01, 0.95)
    df["Opt_Out_Probability"] = base

    # Convert to a predicted purchase probability
    df["Predicted_Purchase_Prob"] = 1 - df["Opt_Out_Probability"]

    # Demand and spend aggregates at row level (can be aggregated in the dashboard)
    df["Predicted_Demand_Units"] = rng.integers(1, 4, size=n_rows)
    df["Unit_Price"] = rng.uniform(40, 180, size=n_rows)
    df["Projected_Spend"] = df["Predicted_Demand_Units"] * df["Unit_Price"] * df[
        "Predicted_Purchase_Prob"
    ]

    return df


__all__ = ["load_feature_table"]

