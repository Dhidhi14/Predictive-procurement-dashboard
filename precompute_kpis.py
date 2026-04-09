import pandas as pd
import numpy as np
import os
import gc

def precompute():
    
    # Relative Path
    input_path = "new/master_data/master_data.csv"
    output_path = "resource/summary_kpis.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("File exists:", os.path.exists(input_path))

    print(f"Starting memory-optimized pre-computation on {input_path}...")
    
    # Define memory-efficient dtypes
    dtype_dict = {
        "College": "category",
        "Year": "category",
        "Semester": "category",
        "Department": "category",
        "ebook_ind": "float32",
        "retail_new": "float32",
        "will_buy": "float32"
    }
    
    # We aggregate in chunks to save memory
    chunk_size = 750000 
    final_summary = None
    
    # Define columns to keep for aggregation
    cols = list(dtype_dict.keys())
    
    try:
        for i, chunk in enumerate(pd.read_csv(input_path, usecols=cols, dtype=dtype_dict, chunksize=chunk_size)):
            # Filter out year '21' or 2021 data
            # Assumes Year is string-like ('21', '2021', 21, 2021)
            # Need to convert to string to strictly check or numeric
            chunk["Year"] = chunk["Year"].astype(str)
            chunk = chunk[~chunk["Year"].isin(["21", "2021", "21.0"])]
            
            # Filter Department to exclude any obvious invalid ones if any, or dropna
            chunk = chunk.dropna(subset=["Department"])

            # Processing at chunk level
            chunk["Spend"] = chunk["retail_new"].fillna(0)
            chunk["Format"] = np.where(chunk["ebook_ind"] == 1.0, "Digital", "Physical")
            chunk["Format"] = chunk["Format"].astype("category")
            
            # Group by categories
            summary = chunk.groupby(["College", "Year", "Semester", "Department", "Format"], observed=True).agg(
                Total_Spend=("Spend", "sum"),
                Book_Count=("Spend", "count")
            ).reset_index()
            
            # Incremental Merge: Merge with existing final_summary to keep RAM constant
            if final_summary is None:
                final_summary = summary
            else:
                final_summary = pd.concat([final_summary, summary], ignore_index=True)
                # Re-aggregate to keep final_summary compact
                final_summary = final_summary.groupby(["College", "Year", "Semester", "Department", "Format"], observed=True).sum().reset_index()
            
            print(f"Processed chunk {i+1} ({ (i+1)*chunk_size:,} rows) ...")
            
            # Frequent garbage collection is key for large CSVs
            del chunk
            del summary
            gc.collect()
            
        # Save to CSV
        if final_summary is not None:
            final_summary.to_csv(output_path, index=False)
            print(f"Pre-computation complete. Saved to {output_path}")
            print(f"Total records processed: {final_summary['Book_Count'].sum():,}")
        else:
            print("No data processed.")
        
    except Exception as e:
        print(f"Error during pre-computation: {e}")

if __name__ == "__main__":
    precompute()
