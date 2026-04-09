import pandas as pd
import os

def aggregate_sentiment(
    input_path: str = "new/master_data/Training_Data_Clean.csv",
    output_path: str = "resource/book_sentiment.csv"
):
    """
    Aggregates student sentiment/rating data at the Book Title level
    to be joined into the main procurement dataset.
    """
    print(f"Reading sentiment data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Normalize book titles for joining
    df["Book_Title"] = df["Book_Title"].str.upper().str.strip()
    
    # List of score columns to aggregate
    score_cols = [
        "Overall_Rating", 
        "How easy was this textbook to understand?_score",
        "How would you rate the conceptual clarity of the book?_score",
        "How engaging and interactive were the examples and exercises?_score",
        "How helpful were the visuals, diagrams, and illustrations?_score",
        "Overall, how satisfied are you with this textbook?_score",
        "Would you recommend this book to other students?_score",
        "How would you rate the value for money of this textbook?_score",
        "Was the book affordable given your financial situation?_score",
        "How useful was this book for scoring well in exams?_score",
        "How likely are you to use this book again in future courses?_score",
        "avg_rating"
    ]
    
    # Filter only available score columns
    available_cols = [c for c in score_cols if c in df.columns]
    
    print(f"Aggregating {len(available_cols)} score columns...")
    
    # Group by Book_Title and take the mean
    sentiment_df = df.groupby("Book_Title", as_index=False)[available_cols].mean()
    
    # Rename columns for clarity in the main ETL
    rename_map = {
        "Overall_Rating": "Book_Rating",
        "How easy was this textbook to understand?_score": "Sentiment_Understandability",
        "How would you rate the value for money of this textbook?_score": "Sentiment_Value_For_Money",
        "How useful was this book for scoring well in exams?_score": "Sentiment_Exam_Utility",
        "avg_rating": "Sentiment_Avg_Rating"
    }
    
    # Apply renames to available columns
    final_rename_map = {k: v for k, v in rename_map.items() if k in available_cols}
    sentiment_df = sentiment_df.rename(columns=final_rename_map)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving aggregated sentiment to {output_path}...")
    sentiment_df.to_csv(output_path, index=False)
    print("Success!")

if __name__ == "__main__":
    aggregate_sentiment()
