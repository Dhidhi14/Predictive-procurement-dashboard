import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from etl_pipeline import load_feature_table

def evaluate_model():
    print("==== Initializing Model Evaluation ====")
    print("Loading data via ETL Pipeline (5% sample)...")
    df = load_feature_table()
    
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
    
    df_clean = df.dropna(subset=features + [target])
    
    if df_clean.empty or len(df_clean[target].unique()) < 2:
        print("Not enough varied data to train/evaluate.")
        return
        
    X = df_clean[features]
    y = df_clean[target]
    
    print(f"Total valid samples: {len(X):,}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training split: {len(X_train):,} | Validation split: {len(X_test):,}")
    print("Training RandomForestClassifier (n_estimators=50, max_depth=5)...")
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    print("Evaluating against test set...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print(f"Accuracy Score: {acc * 100:.2f}%")
    print(f"F1-Score:       {f1:.4f}")
    print("=" * 50)
    print("\nConfusion Matrix:")
    
    cm_df = pd.DataFrame(cm, index=["Actual Opt-Out (0)", "Actual Buy (1)"], columns=["Pred Opt-Out (0)", "Pred Buy (1)"])
    print(cm_df)
    
    print("\n" + "-" * 50)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()
