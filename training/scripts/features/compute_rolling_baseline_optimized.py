import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import sys
from tqdm import tqdm

def compute_rolling_baseline_optimized(input_file, window_sizes=[5, 10], separate_files=True):
    """
    Compute rolling baseline averages for all coordinate features in a CSV file.
    Optimized version that avoids DataFrame fragmentation and can output separate files.
    
    Parameters:
    - input_file: Path to input CSV file
    - window_sizes: List of window sizes for rolling averages (default [5, 10])
    - separate_files: If True, create separate files for each window size
    
    Returns:
    - List of output file paths if successful, empty list otherwise
    """
    
    try:
        # Read the CSV file
        print(f"\nReading: {input_file}")
        df = pd.read_csv(input_file)
        
        # Check if file is empty
        if len(df) == 0:
            print(f"  ⚠️  Skipping empty file")
            return []
        
        # Get all feature columns (those ending with _x, _y, or _z)
        feature_cols = []
        for col in df.columns:
            if col.startswith('feat_') and (col.endswith('_x') or col.endswith('_y') or col.endswith('_z')):
                # Make sure it's a base feature, not a derived one
                if not any(suffix in col for suffix in ['diff', 'rb', 'tot']):
                    feature_cols.append(col)
        
        print(f"  Found {len(feature_cols)} feature columns to process")
        
        input_path = Path(input_file)
        output_files = []
        
        # Process each window size separately
        for window in window_sizes:
            print(f"  Computing rolling baseline with window size {window}...")
            
            # Create dictionaries to store new columns
            new_columns = {}
            
            # Process all feature columns for this window size
            for feat_col in tqdm(feature_cols, desc=f"    Window {window}", leave=False):
                # Compute rolling mean
                rb_col_name = f"{feat_col}_rb{window}"
                rolling_mean = df[feat_col].rolling(window=window, min_periods=1).mean()
                new_columns[rb_col_name] = rolling_mean
                
                # Compute difference from rolling baseline
                diff_col_name = f"{feat_col}_rb{window}_diff"
                new_columns[diff_col_name] = df[feat_col] - rolling_mean
            
            # Create new DataFrame with all columns at once (avoids fragmentation)
            if separate_files:
                # Create a copy of original data with new columns
                df_output = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
                
                # Create output filename for this window size
                output_filename = input_path.stem + f"-rb{window}" + input_path.suffix
                output_path = input_path.parent / output_filename
                
                # Save the file
                print(f"  Saving to: {output_path}")
                df_output.to_csv(output_path, index=False)
                
                # Report file sizes
                input_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
                output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"  ✓ Complete! File size: {input_size:.1f}MB → {output_size:.1f}MB")
                
                output_files.append(str(output_path))
            else:
                # Add columns to main dataframe (for combined output)
                df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
        
        # If not separate files, save combined output
        if not separate_files:
            output_filename = input_path.stem + "-rb" + input_path.suffix
            output_path = input_path.parent / output_filename
            
            print(f"  Saving combined file to: {output_path}")
            df.to_csv(output_path, index=False)
            
            # Report file sizes
            input_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
            output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"  ✓ Complete! File size: {input_size:.1f}MB → {output_size:.1f}MB")
            
            output_files.append(str(output_path))
        
        print(f"  Added {len(new_columns)} new columns per window")
        return output_files
        
    except Exception as e:
        print(f"  ✗ Error processing {input_file}: {str(e)}")
        return []

def process_directory(root_directory=".", window_sizes=[5, 10], separate_files=True):
    """
    Process all CSV files in a directory and its subdirectories.
    
    Parameters:
    - root_directory: Root directory to search for CSV files
    - window_sizes: List of window sizes for rolling averages
    - separate_files: If True, create separate files for each window size
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
    print(f"Output mode: {'Separate files per window' if separate_files else 'Combined file'}")
    print("="*80)
    
    # Track statistics
    successful = 0
    failed = 0
    skipped = 0
    total_files_created = 0
    
    # Process each file
    for csv_file in csv_files:
        result = compute_rolling_baseline_optimized(csv_file, window_sizes, separate_files)
        if result:
            successful += 1
            total_files_created += len(result)
        elif len(result) == 0 and os.path.exists(csv_file):
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
    print(f"Total output files created: {total_files_created}")
    
    if separate_files:
        print(f"\nNew files created with suffixes:")
        for window in window_sizes:
            print(f"  - *-rb{window}.csv: Contains rb{window} and rb{window}_diff features")
    else:
        print(f"\nNew files created with suffix: -rb")
        print(f"Each file contains {len(window_sizes) * 2} new columns per coordinate:")
        for window in window_sizes:
            print(f"  - *_rb{window}: Rolling average with window size {window}")
            print(f"  - *_rb{window}_diff: Difference from rolling average")

def process_single_file(filepath, window_sizes=[5, 10], separate_files=True):
    """
    Process a single CSV file.
    
    Parameters:
    - filepath: Path to the CSV file
    - window_sizes: List of window sizes for rolling averages
    - separate_files: If True, create separate files for each window size
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        return
    
    print(f"\nProcessing single file: {filepath}")
    print(f"Window sizes: {window_sizes}")
    print(f"Output mode: {'Separate files per window' if separate_files else 'Combined file'}")
    print("="*80)
    
    result = compute_rolling_baseline_optimized(filepath, window_sizes, separate_files)
    
    if result:
        print(f"\n✅ Successfully created {len(result)} file(s):")
        for file in result:
            print(f"  - {file}")
    else:
        print(f"\n❌ Failed to process file")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Process directory: python compute_rolling_baseline_optimized.py <directory> [window_sizes] [--combined]")
        print("  Process single file: python compute_rolling_baseline_optimized.py <file.csv> [window_sizes] [--combined]")
        print("\nExamples:")
        print("  python compute_rolling_baseline_optimized.py . (creates separate rb5 and rb10 files)")
        print("  python compute_rolling_baseline_optimized.py . 5,10,20 (creates rb5, rb10, rb20 files)")
        print("  python compute_rolling_baseline_optimized.py . --combined (creates single rb file)")
        print("  python compute_rolling_baseline_optimized.py e25/e25-baseline.csv")
        sys.exit(1)
    
    target = sys.argv[1]
    
    # Parse window sizes if provided
    window_sizes = [5, 10]  # default
    separate_files = True  # default is to create separate files
    
    # Check for --combined flag
    if '--combined' in sys.argv:
        separate_files = False
        sys.argv.remove('--combined')
    
    # Parse window sizes
    if len(sys.argv) > 2:
        try:
            window_sizes = [int(x.strip()) for x in sys.argv[2].split(',')]
        except ValueError:
            print("Error: Window sizes must be comma-separated integers (e.g., 5,10,20)")
            sys.exit(1)
    
    # Check if target is a file or directory
    if os.path.isfile(target):
        process_single_file(target, window_sizes, separate_files)
    elif os.path.isdir(target):
        process_directory(target, window_sizes, separate_files)
    else:
        print(f"Error: '{target}' is not a valid file or directory.")
        sys.exit(1) 