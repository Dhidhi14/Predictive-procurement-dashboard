"""
Feature engineering and ML models for the Predictive Procurement Analytics project.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_and_predict(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train a RandomForest on the dataset to predict Actual_Purchase_Flag.
    Returns:
        df: Updated dataframe with Predicted_Purchase_Prob.
        fi: DataFrame with feature importances.
    """
    if df.empty:
        return df, pd.DataFrame(columns=["Feature", "Importance"])
        
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
    
    train_df = df.dropna(subset=features + [target])
    
    if len(train_df) < 50:
        default_fi = pd.DataFrame({"Feature": features, "Importance": [1.0/len(features)]*len(features)})
        return df, default_fi
        
    X = train_df[features]
    y = train_df[target]
    
    # We predict the probability of purchase (class 1)
    if len(y.unique()) < 2:
        df["Predicted_Purchase_Prob"] = 1.0
        default_fi = pd.DataFrame({"Feature": features, "Importance": [1.0/len(features)]*len(features)})
        return df, default_fi
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    
    # Predict
    X_all = df[features].fillna(df[features].median())
    
    # ensure we predict class 1
    class_1_idx = 1 if len(clf.classes_) > 1 and clf.classes_[1] == 1 else 0
    probs = clf.predict_proba(X_all)[:, class_1_idx]
    
    df["Predicted_Purchase_Prob"] = probs
    # Recalculate Projected Spend with new probs
    df["Projected_Spend"] = df["Predicted_Demand_Units"] * df["Unit_Price"] * df["Predicted_Purchase_Prob"]
    
    fi = pd.DataFrame({
        "Feature": features,
        "Importance": clf.feature_importances_
    })
    
    return df, fi

__all__ = ["train_and_predict"]
