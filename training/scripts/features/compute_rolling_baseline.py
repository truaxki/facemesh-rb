import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import sys
from tqdm import tqdm

def compute_rolling_baseline(input_file, window_sizes=[5, 10], output_suffix="-rb"):
    """
    Compute rolling baseline averages for all coordinate features in a CSV file.
    
    Parameters:
    - input_file: Path to input CSV file
    - window_sizes: List of window sizes for rolling averages (default [5, 10])
    - output_suffix: Suffix to add before extension in output filename (default "-rb")
    
    Returns:
    - Path to output file if successful, None otherwise
    """
    
    try:
        # Read the CSV file
        print(f"\nReading: {input_file}")
        df = pd.read_csv(input_file)
        
        # Check if file is empty
        if len(df) == 0:
            print(f"  ⚠️  Skipping empty file")
            return None
        
        # Get all feature columns (those ending with _x, _y, or _z)
        feature_cols = []
        for col in df.columns:
            if col.startswith('feat_') and (col.endswith('_x') or col.endswith('_y') or col.endswith('_z')):
                # Make sure it's a base feature, not a derived one
                if not any(suffix in col for suffix in ['diff', 'rb', 'tot']):
                    feature_cols.append(col)
        
        print(f"  Found {len(feature_cols)} feature columns to process")
        
        # Process each window size
        new_columns_count = 0
        
        for window in window_sizes:
            print(f"  Computing rolling baseline with window size {window}...")
            
            # Process each feature column
            for feat_col in tqdm(feature_cols, desc=f"    Window {window}", leave=False):
                # Compute rolling mean
                rb_col_name = f"{feat_col}_rb{window}"
                df[rb_col_name] = df[feat_col].rolling(window=window, min_periods=1).mean()
                
                # Compute difference from rolling baseline
                diff_col_name = f"{feat_col}_rb{window}_diff"
                df[diff_col_name] = df[feat_col] - df[rb_col_name]
                
                new_columns_count += 2
        
        print(f"  Added {new_columns_count} new columns")
        
        # Create output filename
        input_path = Path(input_file)
        output_filename = input_path.stem + output_suffix + input_path.suffix
        output_path = input_path.parent / output_filename
        
        # Save the enhanced dataframe
        print(f"  Saving to: {output_path}")
        df.to_csv(output_path, index=False)
        
        # Report file sizes
        input_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"  ✓ Complete! File size: {input_size:.1f}MB → {output_size:.1f}MB")
        
        return str(output_path)
        
    except Exception as e:
        print(f"  ✗ Error processing {input_file}: {str(e)}")
        return None

def process_directory(root_directory=".", window_sizes=[5, 10], pattern="*.csv"):
    """
    Process all CSV files in a directory and its subdirectories.
    
    Parameters:
    - root_directory: Root directory to search for CSV files
    - window_sizes: List of window sizes for rolling averages
    - pattern: File pattern to match (default "*.csv")
    """
    
    # Find all CSV files recursively
    csv_files = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.csv') and '-rb' not in file:  # Skip already processed files
                # Skip non-facemesh files
                if 'self-reported' in file.lower():
                    continue
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print(f"No CSV files found in {root_directory} or its subdirectories.")
        return
    
    print(f"\nFound {len(csv_files)} CSV files to process")
    print(f"Window sizes: {window_sizes}")
    print("="*80)
    
    # Track statistics
    successful = 0
    failed = 0
    skipped = 0
    
    # Process each file
    for csv_file in csv_files:
        result = compute_rolling_baseline(csv_file, window_sizes)
        if result:
            successful += 1
        elif result is None:
            skipped += 1
        else:
            failed += 1
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Successfully processed: {successful} files")
    print(f"Skipped (empty): {skipped} files")
    print(f"Failed: {failed} files")
    print(f"\nNew files created with suffix: -rb")
    print(f"Each file now contains {len(window_sizes) * 2} new columns per coordinate:")
    for window in window_sizes:
        print(f"  - *_rb{window}: Rolling average with window size {window}")
        print(f"  - *_rb{window}_diff: Difference from rolling average")

def process_single_file(filepath, window_sizes=[5, 10]):
    """
    Process a single CSV file.
    
    Parameters:
    - filepath: Path to the CSV file
    - window_sizes: List of window sizes for rolling averages
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        return
    
    print(f"\nProcessing single file: {filepath}")
    print(f"Window sizes: {window_sizes}")
    print("="*80)
    
    result = compute_rolling_baseline(filepath, window_sizes)
    
    if result:
        print(f"\n✅ Successfully created: {result}")
    else:
        print(f"\n❌ Failed to process file")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Process directory: python compute_rolling_baseline.py <directory> [window_sizes]")
        print("  Process single file: python compute_rolling_baseline.py <file.csv> [window_sizes]")
        print("\nExamples:")
        print("  python compute_rolling_baseline.py .")
        print("  python compute_rolling_baseline.py . 5,10,20")
        print("  python compute_rolling_baseline.py e25/e25-baseline.csv")
        print("  python compute_rolling_baseline.py e25/e25-baseline.csv 5,10")
        sys.exit(1)
    
    target = sys.argv[1]
    
    # Parse window sizes if provided
    window_sizes = [5, 10]  # default
    if len(sys.argv) > 2:
        try:
            window_sizes = [int(x.strip()) for x in sys.argv[2].split(',')]
        except ValueError:
            print("Error: Window sizes must be comma-separated integers (e.g., 5,10,20)")
            sys.exit(1)
    
    # Check if target is a file or directory
    if os.path.isfile(target):
        process_single_file(target, window_sizes)
    elif os.path.isdir(target):
        process_directory(target, window_sizes)
    else:
        print(f"Error: '{target}' is not a valid file or directory.")
        sys.exit(1) 