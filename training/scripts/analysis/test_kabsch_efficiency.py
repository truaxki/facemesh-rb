import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from pathlib import Path
import sys

def get_nose_cheeks_features(df, feature_type='rb5_rel_mag'):
    """Get nose and cheeks features."""
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
    
    return feature_cols

def extract_session_type(filename):
    """Extract session type from filename."""
    # e1-baseline-rb5-rel.csv -> baseline
    # e1-session1-rb5-rel.csv -> session1
    parts = filename.stem.split('-')
    if 'baseline' in parts:
        return 'baseline'
    elif 'session' in parts[1]:
        return parts[1]  # session1, session2, etc.
    else:
        return parts[1]  # a, b, c, etc.

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_kabsch_efficiency.py <participant_dir>")
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
        print(f"  - {file.name}: {len(df)} frames, session: {session_type}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal frames: {len(combined_df)}")
    print(f"Session types: {list(session_counts.keys())}")
    print(f"Frames per session: {session_counts}")
    
    # Test both feature types
    results = {}
    
    for feature_type in ['rb5_diff', 'rb5_rel_mag']:
        print(f"\n{'='*80}")
        print(f"Testing {feature_type} features")
        print(f"{'='*80}")
        
        # Get nose+cheeks features
        features = get_nose_cheeks_features(combined_df, feature_type)
        print(f"Found {len(features)} features")
        
        if len(features) == 0:
            print("No features found!")
            continue
        
        print(f"Sample features: {features[:3]}")
        
        # Prepare data
        X = combined_df[features].values
        y = combined_df['session_type']
        
        # Convert session types to numeric
        unique_sessions = sorted(y.unique())
        session_map = {s: i for i, s in enumerate(unique_sessions)}
        y_numeric = y.map(session_map).values
        
        print(f"\nData shape: X={X.shape}, y={y_numeric.shape}")
        print(f"Classes: {unique_sessions}")
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(unique_sessions),
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42
        )
        
        # Cross-validation
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y_numeric, cv=kf, scoring='accuracy')
        
        results[feature_type] = {
            'accuracy': scores.mean() * 100,
            'std': scores.std() * 100,
            'num_features': len(features),
            'efficiency': (scores.mean() * 100) / len(features)
        }
        
        print(f"\nResults:")
        print(f"  Accuracy: {results[feature_type]['accuracy']:.1f}% ± {results[feature_type]['std']:.1f}%")
        print(f"  Features: {results[feature_type]['num_features']}")
        print(f"  Efficiency: {results[feature_type]['efficiency']:.2f}% per feature")
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Feature Type':<20} {'Accuracy':<20} {'Features':<15} {'Efficiency':<20}")
    print("-"*75)
    
    for feat_type, res in results.items():
        print(f"{feat_type:<20} {res['accuracy']:.1f}% ± {res['std']:.1f}%{'':<7} "
              f"{res['num_features']:<15} {res['efficiency']:.2f}% per feature")
    
    # Calculate improvement
    if 'rb5_diff' in results and 'rb5_rel_mag' in results:
        acc_improvement = results['rb5_rel_mag']['accuracy'] - results['rb5_diff']['accuracy']
        print(f"\nKabsch alignment accuracy change: {acc_improvement:+.1f}%")
        print(f"Feature reduction: {results['rb5_rel_mag']['num_features']} vs {results['rb5_diff']['num_features']} "
              f"({results['rb5_rel_mag']['num_features']/results['rb5_diff']['num_features']*100:.0f}% of original)")

if __name__ == "__main__":
    main() 