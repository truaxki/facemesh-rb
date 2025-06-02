"""
Diagnostic script to test facial cluster functionality and feature parsing
"""

import pandas as pd
import sys
from pathlib import Path

# Test imports
try:
    from data_loader_template import FacemeshDataLoader
    from facial_clusters import (
        FACIAL_CLUSTERS, CLUSTER_GROUPS, EXPRESSION_CLUSTERS,
        get_cluster_indices, get_group_indices, 
        get_all_cluster_names, get_all_group_names
    )
    print("✓ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test cluster definitions
print(f"\n=== CLUSTER TESTING ===")
print(f"Total facial clusters: {len(FACIAL_CLUSTERS)}")
print(f"Total cluster groups: {len(CLUSTER_GROUPS)}")
print(f"Total expression clusters: {len(EXPRESSION_CLUSTERS)}")

# Test specific cluster
test_cluster = 'lipsUpperOuter'
indices = get_cluster_indices(test_cluster)
print(f"\n{test_cluster} landmarks: {indices}")
print(f"Count: {len(indices)}")

# Test group indices
test_group = 'mouth'
group_indices = get_group_indices(test_group)
print(f"\n{test_group} group landmarks: {group_indices}")
print(f"Count: {len(group_indices)}")

# Test data loader
loader = FacemeshDataLoader(window_size=5)
print(f"\nData loader initialized:")
print(f"  Window size: {loader.window_size}")
print(f"  Data root: {loader.data_root}")

# Test loading a small sample
print(f"\n=== DATA LOADING TEST ===")
try:
    df_test = loader.load_subject_data("e1", "baseline")
    if not df_test.empty:
        print(f"✓ Successfully loaded data: {len(df_test)} rows, {len(df_test.columns)} columns")
        
        # Check what feature columns exist
        feature_cols = [col for col in df_test.columns if col.startswith('feat_')]
        base_cols = [col for col in feature_cols if not ('_rb' in col)]
        rb_cols = [col for col in feature_cols if f'_rb{loader.window_size}' in col and not col.endswith('_diff')]
        diff_cols = [col for col in feature_cols if col.endswith(f'_rb{loader.window_size}_diff')]
        
        print(f"  Base features: {len(base_cols)}")
        print(f"  Rolling baseline features: {len(rb_cols)}")
        print(f"  Difference features: {len(diff_cols)}")
        
        # Test feature extraction for mouth cluster
        print(f"\n=== FEATURE EXTRACTION TEST ===")
        
        def test_cluster_features(df, cluster_name, feature_types=['diff']):
            """Test version of get_cluster_features with debugging"""
            landmark_indices = get_cluster_indices(cluster_name)
            print(f"Testing {cluster_name} with landmarks: {landmark_indices}")
            
            selected_cols = []
            metadata_cols = ['Subject Name', 'Test Name', 'Time (s)', 'Session']
            selected_cols.extend([col for col in metadata_cols if col in df.columns])
            
            missing_features = []
            found_features = []
            
            for landmark_idx in landmark_indices:
                for feature_type in feature_types:
                    for axis in ['x', 'y', 'z']:
                        if feature_type == 'diff':
                            col_name = f'feat_{landmark_idx}_{axis}_rb{loader.window_size}_diff'
                        elif feature_type == 'rb':
                            col_name = f'feat_{landmark_idx}_{axis}_rb{loader.window_size}'
                        else:  # base
                            col_name = f'feat_{landmark_idx}_{axis}'
                        
                        if col_name in df.columns:
                            selected_cols.append(col_name)
                            found_features.append(col_name)
                        else:
                            missing_features.append(col_name)
            
            print(f"  Found features: {len(found_features)}")
            print(f"  Missing features: {len(missing_features)}")
            if missing_features[:5]:  # Show first 5 missing
                print(f"  Example missing: {missing_features[:5]}")
            if found_features[:5]:  # Show first 5 found
                print(f"  Example found: {found_features[:5]}")
            
            return df[selected_cols] if selected_cols else pd.DataFrame()
        
        # Test mouth cluster
        mouth_result = test_cluster_features(df_test, 'lipsUpperOuter', ['diff'])
        print(f"Mouth cluster result: {len(mouth_result.columns)} columns")
        
        # Test iris clusters (the problematic ones)
        iris_result = test_cluster_features(df_test, 'rightEyeIris', ['diff'])
        print(f"Right eye iris result: {len(iris_result.columns)} columns")
        
    else:
        print("❌ Failed to load test data")
        
except Exception as e:
    print(f"❌ Data loading error: {e}")

print("\n=== DIAGNOSTIC COMPLETE ===") 