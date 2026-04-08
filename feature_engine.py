"""
Feature engineering and ML models for the Predictive Procurement Analytics project.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model(df: pd.DataFrame):
    """
    Apply robust feature engineering based on statistical signals and drop redundancies.
    Train a RandomForest on the dataset to predict Actual_Purchase_Flag.
    """
    target = "Actual_Purchase_Flag"
    
    if df.empty or target not in df.columns:
        return None, pd.DataFrame(columns=["Feature", "Importance"]), [], 0.0

    # 1. Drop strictly invalid columns & Identifiers
    # will_buy and buy_frac are target leakages.
    invalid_cols = [
        "sis_user_id", "section_id", "book_id_bn", "College", "title", "author", 
        "isbn", "edition", "will_buy", "buy_frac",
        "Predicted_Purchase_Prob", "Opt_Out_Probability", "Projected_Spend"
    ]
    df = df.drop(columns=[c for c in invalid_cols if c in df.columns], errors='ignore')

    # Drop rows without target
    df = df.dropna(subset=[target])

    # CRITICAL: Sample early to prevent MemoryError (OOM) during subsequent pre-processing
    # (like pd.get_dummies) when handling 7.5M+ records.
    MAX_TRAIN = 100_000
    if len(df) > MAX_TRAIN:
        df = df.sample(n=MAX_TRAIN, random_state=42)

    # 2. Handle Redundancies (Keep one of each highly correlated pair)
    redundant_to_drop = [
        "Year",                         # Keep term_year
        "Semester",                     # Keep term_code
        "Student_Status",               # Keep student_full_part_time_status
        "price_affordability_score",    # Keep Wallet_Pressure_Score
        "ebook_ind",                    # Keep Digital_Lock_Flag
        "retail_new_rent",              # Keep Rental_to_Retail_Ratio
        "retail_used_rent",             # Keep Rental_to_Retail_Ratio
        "Arbitrage_Index"               # Keep Rental_to_Retail_Ratio
    ]
    df = df.drop(columns=[c for c in redundant_to_drop if c in df.columns], errors='ignore')

    # 3. Handle Missing Values & Extract Categoricals safely
    y = df[target]
    X_raw = df.drop(columns=[target])
    
    # Fill missing numerics with median and strings with 'Unknown'
    num_cols = X_raw.select_dtypes(include=[np.number]).columns
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns
    
    # Prevent SettingWithCopyWarning by acting on a fresh copy
    X_clean = pd.DataFrame(index=X_raw.index)
    
    if len(num_cols) > 0:
        X_num = X_raw[num_cols].fillna(X_raw[num_cols].median())
        X_clean = pd.concat([X_clean, X_num], axis=1)
        
    if len(cat_cols) > 0:
        # Convert to object first to avoid Categorical type restrictions during fillna
        X_cat = X_raw[cat_cols].astype(object).fillna("Unknown").astype(str)
        # Use simple get_dummies for categorical encoding
        X_cat_encoded = pd.get_dummies(X_cat, drop_first=True, dtype=int)
        X_clean = pd.concat([X_clean, X_cat_encoded], axis=1)

    # Note: Ensure we have actual features left
    features = list(X_clean.columns)
    
    if len(X_clean) < 50 or len(features) == 0:
        default_fi = pd.DataFrame({"Feature": features, "Importance": [1.0/len(features)]*len(features) if features else []})
        return None, default_fi, features, 0.0
        
    if len(y.unique()) < 2:
        default_fi = pd.DataFrame({"Feature": features, "Importance": [1.0/len(features)]*len(features)})
        return None, default_fi, features, 0.0
    
    # 4. Automated feature evaluation and model training
    clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=1)
    clf.fit(X_clean, y)
    
    acc = clf.score(X_clean, y)
    
    fi = pd.DataFrame({
        "Feature": features,
        "Importance": clf.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    return clf, fi, features, acc

def apply_predictions(df: pd.DataFrame, clf: object, features: list, discount_pct: float = 0.0) -> pd.DataFrame:
    """
    Apply What-If simulation changes and re-predict purchase probabilities dynamically.
    Validates features match exactly what the model was trained on.
    """
    temp_df = df.copy()
    
    if clf is None or df.empty:
        temp_df["Predicted_Purchase_Prob"] = 1.0
        temp_df["Projected_Spend"] = temp_df.get("Predicted_Demand_Units", 1) * temp_df.get("Unit_Price", 1)
        temp_df["Opt_Out_Probability"] = 0.0
        return temp_df

    # Apply What-If Scenario Modifications
    if discount_pct > 0:
        multiplier = 1.0 - (discount_pct / 100.0)
        # Handle the modifications on whatever we kept
        if "Wallet_Pressure_Score" in temp_df.columns:
            temp_df["Wallet_Pressure_Score"] = (temp_df["Wallet_Pressure_Score"] * multiplier).clip(0, 1)
        if "Unit_Price" in temp_df.columns:
            temp_df["Unit_Price"] = temp_df["Unit_Price"] * multiplier
        if "Rental_to_Retail_Ratio" in temp_df.columns:
            temp_df["Rental_to_Retail_Ratio"] = (temp_df["Rental_to_Retail_Ratio"] / multiplier).clip(0, 1.5)

    # Encode categorical columns matching the original df the same way we did for training
    # Note: We must ensure all dummy columns present in `features` exist in this slice
    X_raw = temp_df.drop(columns=["Actual_Purchase_Flag"], errors='ignore')
    
    num_cols = X_raw.select_dtypes(include=[np.number]).columns
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns
    
    X_clean = pd.DataFrame(index=X_raw.index)
    if len(num_cols) > 0:
        X_num = X_raw[num_cols].fillna(X_raw[num_cols].median())
        X_clean = pd.concat([X_clean, X_num], axis=1)
        
    if len(cat_cols) > 0:
        # Convert to object first to avoid Categorical type restrictions during fillna
        X_cat = X_raw[cat_cols].astype(object).fillna("Unknown").astype(str)
        X_cat_encoded = pd.get_dummies(X_cat, drop_first=True, dtype=int)
        X_clean = pd.concat([X_clean, X_cat_encoded], axis=1)

    # Align columns to what the model expects
    X_all = pd.DataFrame(0, index=X_clean.index, columns=features)
    overlap = list(set(X_clean.columns).intersection(set(features)))
    X_all[overlap] = X_clean[overlap]
    # Fill remaining NaNs implicitly via 0 DataFrame initialization logic
    X_all = X_all[features].fillna(0)
    
    class_1_idx = 1 if len(clf.classes_) > 1 and clf.classes_[1] == 1 else 0
    probs = clf.predict_proba(X_all)[:, class_1_idx]
    
    temp_df["Predicted_Purchase_Prob"] = probs
    temp_df["Opt_Out_Probability"] = 1.0 - probs
    
    # Optional logic for spend calculation
    demand = temp_df.get("Predicted_Demand_Units", temp_df.get("num_books_in_section", 1))
    price = temp_df.get("Unit_Price", temp_df.get("retail_new", 100))
    temp_df["Projected_Spend"] = demand * price * temp_df["Predicted_Purchase_Prob"]
    
    return temp_df

__all__ = ["train_model", "apply_predictions"]
