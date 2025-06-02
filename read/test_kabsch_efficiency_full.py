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

def extract_session_type(filename):
    """Extract session type from filename."""
    parts = filename.stem.split('-')
    if 'baseline' in parts:
        return 'baseline'
    elif 'session' in parts[1]:
        return parts[1]
    else:
        return parts[1]

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_kabsch_efficiency_full.py <participant_dir>")
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
    
    # Test all three feature types
    results = {}
    feature_types = ['rb5_diff', 'rb5_rel_diff', 'rb5_rel_mag']
    
    for feature_type in feature_types:
        print(f"\n{'='*80}")
        print(f"Testing {feature_type} features")
        print(f"{'='*80}")
        
        # Get nose+cheeks features
        features = get_nose_cheeks_features(combined_df, feature_type)
        print(f"Found {len(features)} features")
        
        if len(features) == 0:
            print("No features found!")
            continue
        
        # Prepare data
        X = combined_df[features].values
        y = combined_df['session_type']
        
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
    print("COMPARISON SUMMARY: Nose+Cheeks Efficiency")
    print(f"{'='*80}")
    
    print(f"\n{'Feature Type':<20} {'Accuracy':<20} {'Features':<15} {'Efficiency':<20}")
    print("-"*75)
    
    for feat_type, res in results.items():
        print(f"{feat_type:<20} {res['accuracy']:.1f}% ± {res['std']:.1f}%{'':<7} "
              f"{res['num_features']:<15} {res['efficiency']:.2f}% per feature")
    
    # Key insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    
    if 'rb5_diff' in results and 'rb5_rel_mag' in results:
        orig_acc = results['rb5_diff']['accuracy']
        kabsch_mag_acc = results['rb5_rel_mag']['accuracy']
        
        print(f"\n1. ULTRA-EFFICIENT: Just 6 magnitude features achieve {kabsch_mag_acc:.1f}% accuracy")
        print(f"   - Original 18 features: {orig_acc:.1f}%")
        print(f"   - Accuracy retention: {kabsch_mag_acc/orig_acc*100:.0f}%")
        print(f"   - Feature reduction: 67% (18 → 6 features)")
        
    if 'rb5_rel_diff' in results:
        print(f"\n2. KABSCH COMPONENTS: 18 differential features achieve {results['rb5_rel_diff']['accuracy']:.1f}%")
        print(f"   - Same feature count as original but position-invariant")
        
    print(f"\n3. EFFICIENCY CHAMPION: rb5_rel_mag provides {results['rb5_rel_mag']['efficiency']:.1f}% accuracy per feature")
    print(f"   - {results['rb5_rel_mag']['efficiency']/results['rb5_diff']['efficiency']:.1f}x more efficient than original features")

if __name__ == "__main__":
    main() 