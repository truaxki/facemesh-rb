import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, mean_squared_error
import os
import sys
from pathlib import Path

# Import facial clusters
from facial_clusters import FACIAL_CLUSTERS

def get_nose_cheeks_features(df, feature_type='rb5_rel_mag'):
    """
    Extract nose and cheeks features based on the efficient cluster discovery.
    
    Args:
        df: DataFrame with features
        feature_type: Type of features to extract
            - 'rb5_diff': Original rolling baseline differential features
            - 'rb5_rel_mag': Kabsch-aligned magnitude features
            - 'rb5_rel_diff': Kabsch-aligned differential features (x,y,z)
    
    Returns:
        List of feature column names
    """
    # The magic 6 landmarks
    nose_landmarks = [1, 2, 98, 327]    # noseTip, noseBottom, corners  
    cheek_landmarks = [205, 425]        # rightCheek, leftCheek
    all_landmarks = nose_landmarks + cheek_landmarks
    
    feature_cols = []
    
    if feature_type == 'rb5_diff':
        # Original differential features (18 total: 6 landmarks × 3 axes)
        for landmark_idx in all_landmarks:
            for axis in ['x', 'y', 'z']:
                col_name = f'feat_{landmark_idx}_{axis}_rb5_diff'
                if col_name in df.columns:
                    feature_cols.append(col_name)
                    
    elif feature_type == 'rb5_rel_mag':
        # Kabsch-aligned magnitude features (6 total: 1 per landmark)
        for landmark_idx in all_landmarks:
            col_name = f'feat_{landmark_idx}_rb5_rel_mag'
            if col_name in df.columns:
                feature_cols.append(col_name)
                
    elif feature_type == 'rb5_rel_diff':
        # Kabsch-aligned differential features (18 total: 6 landmarks × 3 axes)
        for landmark_idx in all_landmarks:
            for axis in ['x', 'y', 'z']:
                col_name = f'feat_{landmark_idx}_{axis}_rb5_rel_diff'
                if col_name in df.columns:
                    feature_cols.append(col_name)
    
    return feature_cols

def get_all_expression_features(df, feature_type='rb5_rel_mag'):
    """
    Extract all expression-related landmark features.
    
    Args:
        df: DataFrame with features
        feature_type: Type of features to extract
    
    Returns:
        List of feature column names
    """
    # All expression clusters
    expression_clusters = ['lips', 'upperLip', 'lowerLip', 'leftEye', 'leftEyebrow', 
                          'leftIris', 'rightEye', 'rightEyebrow', 'rightIris', 
                          'midwayBetweenEyes', 'noseTip', 'noseBottom', 'noseRightCorner', 
                          'noseLeftCorner', 'rightCheek', 'leftCheek']
    
    # Get unique landmarks from all expression clusters
    all_landmarks = set()
    for cluster in expression_clusters:
        if cluster in FACIAL_CLUSTERS:
            all_landmarks.update(FACIAL_CLUSTERS[cluster])
    
    feature_cols = []
    
    if feature_type == 'rb5_rel_mag':
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

def run_classification_test(X, y, feature_name, n_splits=3):
    """
    Run XGBoost classification with cross-validation.
    """
    # Convert string labels to numeric if needed
    if y.dtype == 'object':
        unique_labels = sorted(y.unique())
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_numeric = y.map(label_map)
    else:
        y_numeric = y
    
    # XGBoost parameters (same as in original study)
    params = {
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y_numeric)),
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42
    }
    
    # Create model
    model = xgb.XGBClassifier(**params)
    
    # Cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y_numeric, cv=kf, scoring='accuracy')
    
    return {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
        'num_features': X.shape[1],
        'efficiency': scores.mean() / X.shape[1] * 100  # Accuracy per feature
    }

def analyze_participant(participant_dir, feature_types=['rb5_diff', 'rb5_rel_mag', 'rb5_rel_diff']):
    """
    Analyze a single participant's data with different feature types.
    """
    results = {}
    
    # Find all rb5-rel files
    rel_files = list(Path(participant_dir).glob('*-rb5-rel.csv'))
    
    if not rel_files:
        print(f"  No rb5-rel files found in {participant_dir}")
        return results
    
    # Combine all sessions
    all_data = []
    for file in rel_files:
        df = pd.read_csv(file)
        # Extract session type from filename (e.g., 'a', 'b', 'baseline', etc.)
        session_type = file.stem.split('-')[1]
        df['session_type'] = session_type
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Test each feature type
    for feature_type in feature_types:
        print(f"\n  Testing {feature_type}...")
        
        # Get features
        nose_cheek_features = get_nose_cheeks_features(combined_df, feature_type)
        all_expr_features = get_all_expression_features(combined_df, feature_type)
        
        if not nose_cheek_features:
            print(f"    No {feature_type} features found")
            continue
        
        # Prepare data
        X_nose_cheek = combined_df[nose_cheek_features]
        X_all_expr = combined_df[all_expr_features]
        y = combined_df['session_type']
        
        # Run tests
        nose_cheek_results = run_classification_test(X_nose_cheek, y, 'nose_cheeks')
        all_expr_results = run_classification_test(X_all_expr, y, 'all_expression')
        
        # Store results
        results[feature_type] = {
            'nose_cheeks': {
                'accuracy': f"{nose_cheek_results['mean_accuracy']*100:.1f}% ± {nose_cheek_results['std_accuracy']*100:.1f}%",
                'num_features': nose_cheek_results['num_features'],
                'efficiency': f"{nose_cheek_results['efficiency']:.2f}% per feature"
            },
            'all_expression': {
                'accuracy': f"{all_expr_results['mean_accuracy']*100:.1f}% ± {all_expr_results['std_accuracy']*100:.1f}%",
                'num_features': all_expr_results['num_features'],
                'efficiency': f"{all_expr_results['efficiency']:.3f}% per feature"
            }
        }
    
    return results

def display_comparison_table(results):
    """
    Display results in a formatted comparison table.
    """
    print("\n" + "="*100)
    print("KABSCH-ALIGNED FEATURE ANALYSIS RESULTS")
    print("="*100)
    
    for participant, feature_results in results.items():
        print(f"\nParticipant: {participant}")
        print("-"*100)
        
        # Create comparison table
        print(f"{'Feature Type':<20} {'Cluster':<15} {'Accuracy':<20} {'Features':<10} {'Efficiency':<20}")
        print("-"*100)
        
        for feature_type, cluster_results in feature_results.items():
            for cluster_name, metrics in cluster_results.items():
                print(f"{feature_type:<20} {cluster_name:<15} {metrics['accuracy']:<20} "
                      f"{metrics['num_features']:<10} {metrics['efficiency']:<20}")

def main():
    """
    Main analysis function.
    """
    if len(sys.argv) < 2:
        print("Usage: python analyze_kabsch_features.py <directory>")
        print("Example: python analyze_kabsch_features.py .")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    
    print("="*100)
    print("ANALYZING KABSCH-ALIGNED FEATURES VS ORIGINAL FEATURES")
    print("="*100)
    print(f"\nSearching for participant data in: {root_dir}")
    
    # Find all participant directories
    participant_dirs = []
    for item in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, item)) and item.startswith('e'):
            participant_dirs.append(item)
    
    participant_dirs.sort()
    print(f"Found {len(participant_dirs)} participant directories")
    
    # Analyze each participant
    all_results = {}
    
    for participant in participant_dirs[:5]:  # Test first 5 participants
        print(f"\nAnalyzing {participant}...")
        results = analyze_participant(os.path.join(root_dir, participant))
        if results:
            all_results[participant] = results
    
    # Display results
    display_comparison_table(all_results)
    
    # Summary statistics
    print("\n" + "="*100)
    print("SUMMARY: KABSCH ALIGNMENT IMPACT")
    print("="*100)
    
    # Calculate average improvements
    improvements = []
    for participant, feature_results in all_results.items():
        if 'rb5_diff' in feature_results and 'rb5_rel_mag' in feature_results:
            orig_acc = float(feature_results['rb5_diff']['nose_cheeks']['accuracy'].split('%')[0])
            kabsch_acc = float(feature_results['rb5_rel_mag']['nose_cheeks']['accuracy'].split('%')[0])
            improvement = kabsch_acc - orig_acc
            improvements.append(improvement)
    
    if improvements:
        avg_improvement = np.mean(improvements)
        print(f"\nAverage accuracy change with Kabsch alignment: {avg_improvement:+.1f}%")
        print(f"(Positive values indicate improvement)")
    
    print("\nKEY INSIGHTS:")
    print("- rb5_diff: Original rolling baseline differential features")
    print("- rb5_rel_mag: Kabsch-aligned magnitude features (position-invariant)")
    print("- rb5_rel_diff: Kabsch-aligned differential features (x,y,z components)")
    print("\nThe Kabsch alignment removes rigid head motion while preserving facial deformations.")

if __name__ == "__main__":
    main() 