import pandas as pd
import numpy as np
import os

def precompute():
    input_path = "/home/iiitl/Documents/DATA mining project/new/master_data/master_data.csv"
    output_path = "/home/iiitl/Documents/DATA mining project/resource/summary_kpis.csv"
    
    if not os.path.exists("/home/iiitl/Documents/DATA mining project/resource"):
        os.makedirs("/home/iiitl/Documents/DATA mining project/resource")

    print(f"Starting pre-computation on {input_path}...")
    
    # We aggregate in chunks to save memory
    chunk_size = 500000
    aggregated_results = []
    
    # Define columns to keep for aggregation to save memory
    cols = ["College", "Year", "Semester", "Department", "ebook_ind", "retail_new", "will_buy"]
    
    try:
        for i, chunk in enumerate(pd.read_csv(input_path, usecols=cols, chunksize=chunk_size)):
            # Calculate Projected Spend at the row level
            chunk["Spend"] = chunk["retail_new"].fillna(0) * chunk["will_buy"].fillna(1)
            chunk["Format"] = chunk["ebook_ind"].fillna(0).map(lambda x: "Digital" if x == 1.0 else "Physical")
            
            # Group by categories
            summary = chunk.groupby(["College", "Year", "Semester", "Department", "Format"]).agg(
                Total_Spend=("Spend", "sum"),
                Book_Count=("Spend", "count")
            ).reset_index()
            
            aggregated_results.append(summary)
            print(f"Processed chunk {i+1}...")
            
        # Combine all chunk summaries
        final_summary = pd.concat(aggregated_results).groupby(["College", "Year", "Semester", "Department", "Format"]).sum().reset_index()
        
        # Save to CSV
        final_summary.to_csv(output_path, index=False)
        print(f"Pre-computation complete. Saved to {output_path}")
        print(f"Total records processed: {final_summary['Book_Count'].sum():,}")
        
    except Exception as e:
        print(f"Error during pre-computation: {e}")

if __name__ == "__main__":
    precompute()
