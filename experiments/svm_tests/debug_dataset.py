#!/usr/bin/env python3
"""
DEBUG DATASET STRUCTURE
=======================
Quick script to identify the column structure issue
"""

import pandas as pd
import numpy as np

def debug_dataset(participant_id):
    dataset_path = f"../../training/datasets/{participant_id}_all_sessions_expanded_kabsch.csv"
    
    print(f"🔍 Debugging {participant_id} dataset...")
    df = pd.read_csv(dataset_path)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"First 10 columns: {df.columns[:10].tolist()}")
    
    # Check data types
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    # Find non-numeric columns
    non_numeric = []
    for col in df.columns:
        if df[col].dtype == 'object':
            non_numeric.append(col)
            print(f"Non-numeric column: {col}")
            print(f"  Sample values: {df[col].head().tolist()}")
    
    # Check for target column
    if 'target_session' in df.columns:
        print(f"\nTarget column found: 'target_session'")
        print(f"Unique targets: {df['target_session'].nunique()}")
        print(f"Sample targets: {df['target_session'].head().tolist()}")
    else:
        print(f"\n❌ No 'target_session' column found!")
        print(f"Available columns: {df.columns.tolist()}")

if __name__ == "__main__":
    debug_dataset("e17") 