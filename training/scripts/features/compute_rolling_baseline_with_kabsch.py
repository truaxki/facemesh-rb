import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from data_filters import DataFilters

def extract_3d_points(df, frame_idx, scale_z=True):
    """
    Extract 3D point cloud from a single frame.
    
    Mathematical representation:
    P = [[x_0, y_0, z_0],
         [x_1, y_1, z_1],
         ...
         [x_477, y_477, z_477]]
    
    If scale_z=True:
    z_scaled = z * Face_Depth_cm
    
    Args:
        df: DataFrame with feature columns
        frame_idx: Row index to extract
        scale_z: Whether to scale z coordinates by Face Depth
        
    Returns:
        numpy array of shape (478, 3)
    """
    points = []
    
    # Get Face Depth for this frame if scaling
    face_depth = df.iloc[frame_idx]['Face Depth (cm)'] if scale_z else 1.0
    
    # Extract all 478 landmarks
    for landmark_idx in range(478):
        x = df.iloc[frame_idx][f'feat_{landmark_idx}_x']
        y = df.iloc[frame_idx][f'feat_{landmark_idx}_y']
        z = df.iloc[frame_idx][f'feat_{landmark_idx}_z'] * face_depth
        points.append([x, y, z])
    
    return np.array(points)

def compute_rolling_baseline_points(df, frame_idx, window_size, scale_z=True):
    """
    Compute average 3D point cloud from rolling window.
    
    Mathematical representation:
    Q_baseline = (1/n) * Σ[P_i] for i in [frame_idx-window_size+1, frame_idx]
    
    Args:
        df: DataFrame with feature columns
        frame_idx: Current frame index
        window_size: Size of rolling window (5 or 10)
        scale_z: Whether to scale z coordinates
        
    Returns:
        numpy array of shape (478, 3) representing baseline point cloud
    """
    # Determine window bounds
    start_idx = max(0, frame_idx - window_size + 1)
    end_idx = frame_idx + 1
    
    # Extract points for each frame in window
    window_points = []
    for idx in range(start_idx, end_idx):
        points = extract_3d_points(df, idx, scale_z)
        window_points.append(points)
    
    # Compute average
    baseline_points = np.mean(window_points, axis=0)
    return baseline_points

def compute_kabsch_relative_features(df, window_sizes=[5, 10]):
    """
    Compute Kabsch-aligned relative differences for all frames.
    
    Mathematical process for each frame t:
    1. Extract current frame points: P_t
    2. Extract baseline points: Q_baseline = average of last n frames
    3. Apply Kabsch algorithm: R, t = kabsch(Q_baseline, P_t)
    4. Transform current points: P_aligned = R @ P_t + t
    5. Compute differences: rel_diff = P_aligned - Q_baseline
    6. Compute magnitude: rel_mag = ||rel_diff||
    
    Args:
        df: DataFrame with feature columns
        window_sizes: List of window sizes for rolling baseline
        
    Returns:
        Dictionary of new columns to add
    """
    results = {}
    
    for window in window_sizes:
        print(f"  Computing Kabsch-aligned features for window size {window}...")
        
        # Initialize storage for new features
        rel_diff_features = {f'feat_{i}_{axis}_rb{window}_rel_diff': [] 
                           for i in range(478) 
                           for axis in ['x', 'y', 'z']}
        rel_mag_features = {f'feat_{i}_rb{window}_rel_mag': [] 
                          for i in range(478)}
        
        # Process each frame
        for frame_idx in tqdm(range(len(df)), desc=f"    Processing frames (rb{window})", leave=False):
            # Extract current frame points (with z-scaling)
            P_current = extract_3d_points(df, frame_idx, scale_z=True)
            
            # Get baseline points
            if frame_idx < window:
                # Not enough history, use what we have
                Q_baseline = compute_rolling_baseline_points(df, frame_idx, frame_idx + 1, scale_z=True)
            else:
                Q_baseline = compute_rolling_baseline_points(df, frame_idx, window, scale_z=True)
            
            # Apply Kabsch algorithm (no scaling - just rotation and translation)
            # We align P_current to Q_baseline
            R, t, rmsd = DataFilters.kabsch_algorithm(Q_baseline, P_current)
            
            # Transform current points to align with baseline
            P_aligned = DataFilters.apply_transformation(P_current, R, t)
            
            # Compute relative differences
            rel_diff = P_aligned - Q_baseline
            
            # Store results for each landmark
            for landmark_idx in range(478):
                # Individual components
                rel_diff_features[f'feat_{landmark_idx}_x_rb{window}_rel_diff'].append(rel_diff[landmark_idx, 0])
                rel_diff_features[f'feat_{landmark_idx}_y_rb{window}_rel_diff'].append(rel_diff[landmark_idx, 1])
                rel_diff_features[f'feat_{landmark_idx}_z_rb{window}_rel_diff'].append(rel_diff[landmark_idx, 2])
                
                # Magnitude
                magnitude = np.linalg.norm(rel_diff[landmark_idx])
                rel_mag_features[f'feat_{landmark_idx}_rb{window}_rel_mag'].append(magnitude)
        
        # Add to results
        results.update(rel_diff_features)
        results.update(rel_mag_features)
    
    return results

def process_file_with_kabsch(input_file, window_sizes=[5, 10]):
    """
    Process a single file: compute rolling baseline and Kabsch-aligned features.
    
    Complete mathematical pipeline:
    1. Load original data
    2. Scale z-coordinates: z_scaled = z * Face_Depth_cm
    3. Compute rolling baselines (standard rb features)
    4. For each frame:
       - Extract 3D point cloud (with scaled z)
       - Compute baseline point cloud (average of last n frames)
       - Apply Kabsch alignment to remove rigid motion
       - Compute position-invariant differences
    5. Save all features to new file
    
    Args:
        input_file: Path to input CSV file
        window_sizes: List of window sizes
        
    Returns:
        List of output file paths
    """
    try:
        # Read the CSV file
        print(f"\nReading: {input_file}")
        df = pd.read_csv(input_file)
        
        # Check if file is empty
        if len(df) == 0:
            print(f"  ⚠️  Skipping empty file")
            return []
        
        # First, compute standard rolling baseline features (with scaled z internally)
        print(f"  Computing standard rolling baseline features...")
        
        # Get all feature columns
        feature_cols = []
        for col in df.columns:
            if col.startswith('feat_') and (col.endswith('_x') or col.endswith('_y') or col.endswith('_z')):
                if not any(suffix in col for suffix in ['diff', 'rb', 'tot', 'rel']):
                    feature_cols.append(col)
        
        input_path = Path(input_file)
        output_files = []
        
        # Process each window size
        for window in window_sizes:
            print(f"  Processing window size {window}...")
            
            # Create a copy of the dataframe
            df_output = df.copy()
            
            # Add standard rolling baseline features
            for feat_col in feature_cols:
                # For z-coordinates, scale before computing rolling baseline
                if feat_col.endswith('_z'):
                    scaled_values = df[feat_col] * df['Face Depth (cm)']
                    rolling_mean = scaled_values.rolling(window=window, min_periods=1).mean()
                    df_output[f"{feat_col}_rb{window}"] = rolling_mean
                    df_output[f"{feat_col}_rb{window}_diff"] = scaled_values - rolling_mean
                else:
                    rolling_mean = df[feat_col].rolling(window=window, min_periods=1).mean()
                    df_output[f"{feat_col}_rb{window}"] = rolling_mean
                    df_output[f"{feat_col}_rb{window}_diff"] = df[feat_col] - rolling_mean
            
            # Compute Kabsch-aligned relative features
            print(f"  Computing Kabsch-aligned relative features...")
            kabsch_features = compute_kabsch_relative_features(df, [window])
            
            # Add Kabsch features to dataframe
            for col_name, values in kabsch_features.items():
                df_output[col_name] = values
            
            # Create output filename
            output_filename = input_path.stem + f"-rb{window}-rel" + input_path.suffix
            output_path = input_path.parent / output_filename
            
            # Save the file
            print(f"  Saving to: {output_path}")
            df_output.to_csv(output_path, index=False)
            
            # Report statistics
            original_cols = len(df.columns)
            new_cols = len(df_output.columns)
            print(f"  ✓ Complete! Columns: {original_cols} → {new_cols} (+{new_cols - original_cols})")
            
            output_files.append(str(output_path))
        
        return output_files
        
    except Exception as e:
        print(f"  ✗ Error processing {input_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def process_directory(root_directory=".", window_sizes=[5, 10]):
    """
    Process all CSV files in a directory with Kabsch alignment.
    """
    # Find all CSV files
    csv_files = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.csv') and not any(suffix in file for suffix in ['-rb', '-rel']):
                if 'self-reported' in file.lower():
                    continue
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print(f"No CSV files found in {root_directory}")
        return
    
    print(f"\nFound {len(csv_files)} CSV files to process")
    print(f"Window sizes: {window_sizes}")
    print("="*80)
    
    # Process each file
    successful = 0
    failed = 0
    total_files_created = 0
    
    for csv_file in csv_files:
        result = process_file_with_kabsch(csv_file, window_sizes)
        if result:
            successful += 1
            total_files_created += len(result)
        else:
            failed += 1
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Total output files created: {total_files_created}")
    print(f"\nFiles created with pattern: *-rb{{window}}-rel.csv")

def display_mathematical_summary():
    """
    Display clear mathematical formulation for Jupyter notebook.
    """
    print("""
    ========================================================================
    MATHEMATICAL FORMULATION: Kabsch-Aligned Relative Differences
    ========================================================================
    
    1. Z-COORDINATE SCALING:
       For each landmark i and frame t:
       z_scaled[i,t] = z[i,t] × FaceDepth[t]
    
    2. 3D POINT EXTRACTION:
       P[t] = [[x[0,t], y[0,t], z_scaled[0,t]],
               [x[1,t], y[1,t], z_scaled[1,t]],
               ...
               [x[477,t], y[477,t], z_scaled[477,t]]]
    
    3. ROLLING BASELINE COMPUTATION:
       Q_baseline[t] = (1/n) × Σ P[k] for k ∈ [t-n+1, t]
       where n = window size (5 or 10)
    
    4. KABSCH ALIGNMENT:
       Find optimal R (rotation) and t (translation) such that:
       ||R×P[t] + t - Q_baseline[t]||² is minimized
    
    5. TRANSFORM CURRENT FRAME:
       P_aligned[t] = R×P[t] + t
    
    6. COMPUTE RELATIVE DIFFERENCES:
       rel_diff[i,t] = P_aligned[i,t] - Q_baseline[i,t]
       where i = landmark index
    
    7. COMPUTE MAGNITUDE:
       rel_mag[i,t] = ||rel_diff[i,t]|| = √(Δx² + Δy² + Δz²)
    
    INTERPRETATION:
    - rel_diff represents position-invariant movement
    - Removes global head motion (rotation/translation)
    - Preserves local facial deformations
    - rel_mag quantifies overall movement intensity
    
    ========================================================================
    """)

if __name__ == "__main__":
    # Display mathematical summary
    display_mathematical_summary()
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  Process directory: python compute_rolling_baseline_with_kabsch.py <directory> [window_sizes]")
        print("  Process single file: python compute_rolling_baseline_with_kabsch.py <file.csv> [window_sizes]")
        print("\nExamples:")
        print("  python compute_rolling_baseline_with_kabsch.py .")
        print("  python compute_rolling_baseline_with_kabsch.py . 5,10")
        print("  python compute_rolling_baseline_with_kabsch.py e25/e25-baseline.csv")
        sys.exit(1)
    
    target = sys.argv[1]
    
    # Parse window sizes
    window_sizes = [5, 10]  # default
    if len(sys.argv) > 2:
        try:
            window_sizes = [int(x.strip()) for x in sys.argv[2].split(',')]
        except ValueError:
            print("Error: Window sizes must be comma-separated integers")
            sys.exit(1)
    
    # Process target
    if os.path.isfile(target):
        process_file_with_kabsch(target, window_sizes)
    elif os.path.isdir(target):
        process_directory(target, window_sizes)
    else:
        print(f"Error: '{target}' is not a valid file or directory")
        sys.exit(1) 