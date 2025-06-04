import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from pathlib import Path
import sys
from facial_clusters import FACIAL_CLUSTERS

def get_lips_eyebrows_features(df, feature_type='rb5_rel_mag'):
    """Get lips and eyebrows features."""
    # Get lip landmarks
    lips_landmarks = []
    lips_landmarks.extend(FACIAL_CLUSTERS.get('lipsUpperOuter', []))
    lips_landmarks.extend(FACIAL_CLUSTERS.get('lipsLowerOuter', []))
    lips_landmarks.extend(FACIAL_CLUSTERS.get('lipsUpperInner', []))
    lips_landmarks.extend(FACIAL_CLUSTERS.get('lipsLowerInner', []))
    lips_landmarks = list(set(lips_landmarks))  # Remove duplicates
    
    # Get eyebrow landmarks
    eyebrows_landmarks = []
    eyebrows_landmarks.extend(FACIAL_CLUSTERS.get('rightEyebrowUpper', []))
    eyebrows_landmarks.extend(FACIAL_CLUSTERS.get('rightEyebrowLower', []))
    eyebrows_landmarks.extend(FACIAL_CLUSTERS.get('leftEyebrowUpper', []))
    eyebrows_landmarks.extend(FACIAL_CLUSTERS.get('leftEyebrowLower', []))
    eyebrows_landmarks = list(set(eyebrows_landmarks))  # Remove duplicates
    
    all_landmarks = lips_landmarks + eyebrows_landmarks
    
    print(f"  Lips landmarks ({len(lips_landmarks)}): {lips_landmarks[:5]}...")
    print(f"  Eyebrows landmarks ({len(eyebrows_landmarks)}): {eyebrows_landmarks[:5]}...")
    
    feature_cols = []
    missing_landmarks = []
    
    if feature_type == 'rb5_diff':
        for landmark_idx in all_landmarks:
            found = False
            for axis in ['x', 'y', 'z']:
                col_name = f'feat_{landmark_idx}_{axis}_rb5_diff'
                if col_name in df.columns:
                    feature_cols.append(col_name)
                    found = True
            if not found:
                missing_landmarks.append(landmark_idx)
                    
    elif feature_type == 'rb5_rel_mag':
        for landmark_idx in all_landmarks:
            col_name = f'feat_{landmark_idx}_rb5_rel_mag'
            if col_name in df.columns:
                feature_cols.append(col_name)
            else:
                missing_landmarks.append(landmark_idx)
                
    elif feature_type == 'rb5_rel_diff':
        for landmark_idx in all_landmarks:
            found = False
            for axis in ['x', 'y', 'z']:
                col_name = f'feat_{landmark_idx}_{axis}_rb5_rel_diff'
                if col_name in df.columns:
                    feature_cols.append(col_name)
                    found = True
            if not found:
                missing_landmarks.append(landmark_idx)
    
    if missing_landmarks:
        print(f"  ⚠️ Missing landmarks for {feature_type}: {missing_landmarks}")
    
    return feature_cols

def get_nose_cheeks_features(df, feature_type='rb5_rel_mag'):
    """Get nose and cheeks features for comparison."""
    nose_landmarks = [1, 2, 98, 327]
    cheek_landmarks = [205, 425]
    all_landmarks = nose_landmarks + cheek_landmarks
    
    feature_cols = []
    
    if feature_type == 'rb5_diff':
        for landmark_idx in all_landmarks:
            for axis in ['x', 'y', 'z']:
                col_name = f'feat_{landmark_idx}_{axis}_rb5_diff'
                if col_name in df.columns:
                    feature_cols.append(col_name)
                    
    elif feature_type == 'rb5_rel_mag':
        for landmark_idx in all_landmarks:
            col_name = f'feat_{landmark_idx}_rb5_rel_mag'
            if col_name in df.columns:
                feature_cols.append(col_name)
                
    elif feature_type == 'rb5_rel_diff':
        for landmark_idx in all_landmarks:
            for axis in ['x', 'y', 'z']:
                col_name = f'feat_{landmark_idx}_{axis}_rb5_rel_diff'
                if col_name in df.columns:
                    feature_cols.append(col_name)
    
    return feature_cols

def extract_session_type(filename):
    """Extract session type from filename."""
    parts = filename.stem.split('-')
    if 'baseline' in parts:
        return 'baseline'
    elif 'session' in parts[1]:
        return parts[1]
    else:
        return parts[1]

def test_cluster(df, features, cluster_name, feature_type):
    """Test a specific cluster configuration."""
    if len(features) == 0:
        return None
    
    # Prepare data
    X = df[features].values
    y = df['session_type']
    
    # Convert session types to numeric
    unique_sessions = sorted(y.unique())
    session_map = {s: i for i, s in enumerate(unique_sessions)}
    y_numeric = y.map(session_map).values
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(unique_sessions),
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        verbosity=0
    )
    
    # Cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y_numeric, cv=kf, scoring='accuracy')
    
    return {
        'accuracy': scores.mean() * 100,
        'std': scores.std() * 100,
        'num_features': len(features),
        'efficiency': (scores.mean() * 100) / len(features)
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_kabsch_lips_eyebrows.py <participant_dir>")
        sys.exit(1)
    
    participant_dir = Path(sys.argv[1])
    print(f"\nAnalyzing participant: {participant_dir.name}")
    print("="*80)
    
    # Find all rb5-rel files
    rel_files = list(participant_dir.glob('*-rb5-rel.csv'))
    
    if not rel_files:
        print(f"No rb5-rel files found in {participant_dir}")
        return
    
    print(f"Found {len(rel_files)} files")
    
    # Combine all sessions
    all_data = []
    session_counts = {}
    
    for file in rel_files:
        df = pd.read_csv(file)
        session_type = extract_session_type(file)
        df['session_type'] = session_type
        
        if session_type not in session_counts:
            session_counts[session_type] = 0
        session_counts[session_type] += len(df)
        
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal frames: {len(combined_df)}")
    print(f"Session types: {list(session_counts.keys())}")
    
    # Test all three feature types for both clusters
    results = {}
    feature_types = ['rb5_diff', 'rb5_rel_diff', 'rb5_rel_mag']
    
    for feature_type in feature_types:
        print(f"\n{'='*80}")
        print(f"Testing {feature_type} features")
        print(f"{'='*80}")
        
        # Test lips+eyebrows
        print("\nLips + Eyebrows cluster:")
        lips_eyebrows_features = get_lips_eyebrows_features(combined_df, feature_type)
        print(f"Found {len(lips_eyebrows_features)} features")
        
        if lips_eyebrows_features:
            lips_eyebrows_results = test_cluster(combined_df, lips_eyebrows_features, 
                                               'lips_eyebrows', feature_type)
            if lips_eyebrows_results:
                results[f'{feature_type}_lips_eyebrows'] = lips_eyebrows_results
        
        # Test nose+cheeks for comparison
        print("\nNose + Cheeks cluster (for comparison):")
        nose_cheeks_features = get_nose_cheeks_features(combined_df, feature_type)
        print(f"Found {len(nose_cheeks_features)} features")
        
        if nose_cheeks_features:
            nose_cheeks_results = test_cluster(combined_df, nose_cheeks_features, 
                                             'nose_cheeks', feature_type)
            if nose_cheeks_results:
                results[f'{feature_type}_nose_cheeks'] = nose_cheeks_results
    
    # Display comprehensive comparison
    print(f"\n{'='*100}")
    print("COMPREHENSIVE COMPARISON: Lips+Eyebrows vs Nose+Cheeks")
    print(f"{'='*100}")
    
    print(f"\n{'Cluster':<20} {'Feature Type':<15} {'Accuracy':<20} {'Features':<10} {'Efficiency':<20}")
    print("-"*100)
    
    # Group by feature type for better comparison
    for feature_type in feature_types:
        for cluster in ['lips_eyebrows', 'nose_cheeks']:
            key = f'{feature_type}_{cluster}'
            if key in results:
                res = results[key]
                cluster_display = cluster.replace('_', '+').title()
                print(f"{cluster_display:<20} {feature_type:<15} {res['accuracy']:.1f}% ± {res['std']:.1f}%{'':<7} "
                      f"{res['num_features']:<10} {res['efficiency']:.2f}% per feature")
        print()  # Empty line between feature types
    
    # Key insights
    print(f"{'='*100}")
    print("KEY INSIGHTS: KABSCH ALIGNMENT IMPACT ON DIFFERENT REGIONS")
    print(f"{'='*100}")
    
    # Compare rb5_rel_mag efficiency between clusters
    if 'rb5_rel_mag_lips_eyebrows' in results and 'rb5_rel_mag_nose_cheeks' in results:
        lips_mag = results['rb5_rel_mag_lips_eyebrows']
        nose_mag = results['rb5_rel_mag_nose_cheeks']
        
        print(f"\n1. MAGNITUDE FEATURES (rb5_rel_mag) COMPARISON:")
        print(f"   - Lips+Eyebrows: {lips_mag['accuracy']:.1f}% with {lips_mag['num_features']} features")
        print(f"   - Nose+Cheeks: {nose_mag['accuracy']:.1f}% with {nose_mag['num_features']} features")
        print(f"   - Winner: {'Lips+Eyebrows' if lips_mag['accuracy'] > nose_mag['accuracy'] else 'Nose+Cheeks'}")
    
    # Compare improvement from Kabsch alignment
    if 'rb5_diff_lips_eyebrows' in results and 'rb5_rel_diff_lips_eyebrows' in results:
        orig_lips = results['rb5_diff_lips_eyebrows']['accuracy']
        kabsch_lips = results['rb5_rel_diff_lips_eyebrows']['accuracy']
        improvement_lips = kabsch_lips - orig_lips
        
        print(f"\n2. KABSCH ALIGNMENT IMPROVEMENT (rb5_diff → rb5_rel_diff):")
        print(f"   - Lips+Eyebrows: {orig_lips:.1f}% → {kabsch_lips:.1f}% ({improvement_lips:+.1f}%)")
        
    if 'rb5_diff_nose_cheeks' in results and 'rb5_rel_diff_nose_cheeks' in results:
        orig_nose = results['rb5_diff_nose_cheeks']['accuracy']
        kabsch_nose = results['rb5_rel_diff_nose_cheeks']['accuracy']
        improvement_nose = kabsch_nose - orig_nose
        print(f"   - Nose+Cheeks: {orig_nose:.1f}% → {kabsch_nose:.1f}% ({improvement_nose:+.1f}%)")
        
        if 'improvement_lips' in locals():
            print(f"\n   → Kabsch benefits {'Lips+Eyebrows' if improvement_lips > improvement_nose else 'Nose+Cheeks'} more!")

if __name__ == "__main__":
    main() 