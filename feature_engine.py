"""
Feature engineering and ML models for the Predictive Procurement Analytics project.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(df: pd.DataFrame):
    """
    Train a RandomForest on the dataset to predict Actual_Purchase_Flag.
    Returns:
        clf: Trained RandomForestClassifier
        fi: DataFrame with feature importances
        features: list of feature names used
        acc: accuracy score (float)
    """
    features = [
        "Arbitrage_Index",
        "Wallet_Pressure_Score",
        "Digital_Lock_Flag",
        "Rental_to_Retail_Ratio",
        "family_annual_income",
        "is_rental",
        "has_scholarship"
    ]
    target = "Actual_Purchase_Flag"
    
    if df.empty:
        return None, pd.DataFrame(columns=["Feature", "Importance"]), features, 0.0
        
    train_df = df.dropna(subset=features + [target])
    
    if len(train_df) < 50:
        default_fi = pd.DataFrame({"Feature": features, "Importance": [1.0/len(features)]*len(features)})
        return None, default_fi, features, 0.0
        
    X = train_df[features]
    y = train_df[target]
    
    if len(y.unique()) < 2:
        default_fi = pd.DataFrame({"Feature": features, "Importance": [1.0/len(features)]*len(features)})
        return None, default_fi, features, 0.0
    
    # NOTE: n_jobs=1 (single-threaded) is intentional.
    # n_jobs=-1 spawns one process per CPU core, each holding a full copy of
    # the training data in RAM — that is what previously caused the system crash.
    # Cap training rows to 100k so .fit() memory stays bounded and trains fast.
    MAX_TRAIN = 100_000
    if len(train_df) > MAX_TRAIN:
        train_df = train_df.sample(n=MAX_TRAIN, random_state=42)
        X = train_df[features]
        y = train_df[target]

    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=1)
    clf.fit(X, y)
    
    # Calculate accuracy on the training pool (proxy for model health)
    acc = clf.score(X, y)
    
    fi = pd.DataFrame({
        "Feature": features,
        "Importance": clf.feature_importances_
    })
    
    return clf, fi, features, acc

def apply_predictions(df: pd.DataFrame, clf: object, features: list, discount_pct: float = 0.0) -> pd.DataFrame:
    """
    Apply What-If simulation changes and re-predict purchase probabilities dynamically.
    """
    temp_df = df.copy()
    
    if clf is None or df.empty:
        temp_df["Predicted_Purchase_Prob"] = 1.0
        temp_df["Projected_Spend"] = temp_df["Predicted_Demand_Units"] * temp_df["Unit_Price"]
        temp_df["Opt_Out_Probability"] = 0.0
        return temp_df

    # Apply What-If Scenario Modifications
    if discount_pct > 0:
        multiplier = 1.0 - (discount_pct / 100.0)
        temp_df["Wallet_Pressure_Score"] = (temp_df["Wallet_Pressure_Score"] * multiplier).clip(0, 1)
        temp_df["Unit_Price"] = temp_df["Unit_Price"] * multiplier
        temp_df["Rental_to_Retail_Ratio"] = (temp_df["Rental_to_Retail_Ratio"] / multiplier).clip(0, 1.5)
        temp_df["Arbitrage_Index"] = 1.0 - temp_df["Rental_to_Retail_Ratio"]

    X_all = temp_df[features].fillna(temp_df[features].median())
    
    class_1_idx = 1 if len(clf.classes_) > 1 and clf.classes_[1] == 1 else 0
    probs = clf.predict_proba(X_all)[:, class_1_idx]
    
    temp_df["Predicted_Purchase_Prob"] = probs
    temp_df["Opt_Out_Probability"] = 1.0 - probs
    temp_df["Projected_Spend"] = temp_df["Predicted_Demand_Units"] * temp_df["Unit_Price"] * temp_df["Predicted_Purchase_Prob"]
    
    return temp_df

__all__ = ["train_model", "apply_predictions"]
