import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from pathlib import Path
import sys
import time
from facial_clusters import FACIAL_CLUSTERS

def get_lips_eyebrows_features(df, feature_type='rb5_rel_diff'):
    """Get lips and eyebrows features."""
    lips_landmarks = []
    lips_landmarks.extend(FACIAL_CLUSTERS.get('lipsUpperOuter', []))
    lips_landmarks.extend(FACIAL_CLUSTERS.get('lipsLowerOuter', []))
    lips_landmarks.extend(FACIAL_CLUSTERS.get('lipsUpperInner', []))
    lips_landmarks.extend(FACIAL_CLUSTERS.get('lipsLowerInner', []))
    lips_landmarks = list(set(lips_landmarks))
    
    eyebrows_landmarks = []
    eyebrows_landmarks.extend(FACIAL_CLUSTERS.get('rightEyebrowUpper', []))
    eyebrows_landmarks.extend(FACIAL_CLUSTERS.get('rightEyebrowLower', []))
    eyebrows_landmarks.extend(FACIAL_CLUSTERS.get('leftEyebrowUpper', []))
    eyebrows_landmarks.extend(FACIAL_CLUSTERS.get('leftEyebrowLower', []))
    eyebrows_landmarks = list(set(eyebrows_landmarks))
    
    all_landmarks = lips_landmarks + eyebrows_landmarks
    
    feature_cols = []
    
    if feature_type == 'rb5_rel_diff':
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

def test_model_config(X, y, config_name, params):
    """Test a specific model configuration."""
    start_time = time.time()
    
    # Create model
    model = xgb.XGBClassifier(**params, random_state=42, verbosity=0)
    
    # Cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    
    training_time = time.time() - start_time
    
    return {
        'config': config_name,
        'accuracy': scores.mean() * 100,
        'std': scores.std() * 100,
        'training_time': training_time,
        'params': params
    }

def hyperparameter_search(X, y):
    """Perform grid search for optimal hyperparameters."""
    print("\n🔍 Running hyperparameter search (this may take a few minutes)...")
    
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        random_state=42,
        verbosity=0
    )
    
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X, y)
    search_time = time.time() - start_time
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_ * 100,
        'search_time': search_time,
        'n_combinations': len(grid_search.cv_results_['params'])
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_model_optimization.py <participant_dir>")
        sys.exit(1)
    
    participant_dir = Path(sys.argv[1])
    print(f"\n🚀 MODEL OPTIMIZATION ANALYSIS")
    print(f"Participant: {participant_dir.name}")
    print("="*80)
    
    # Load data
    rel_files = list(participant_dir.glob('*-rb5-rel.csv'))
    if not rel_files:
        print(f"No rb5-rel files found")
        return
    
    all_data = []
    for file in rel_files:
        df = pd.read_csv(file)
        session_type = extract_session_type(file)
        df['session_type'] = session_type
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Get best features (Lips+Eyebrows with rb5_rel_diff)
    features = get_lips_eyebrows_features(combined_df, 'rb5_rel_diff')
    print(f"\nUsing Lips+Eyebrows rb5_rel_diff features: {len(features)} features")
    
    X = combined_df[features].values
    y = combined_df['session_type']
    
    # Convert to numeric
    unique_sessions = sorted(y.unique())
    session_map = {s: i for i, s in enumerate(unique_sessions)}
    y_numeric = y.map(session_map).values
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(unique_sessions)} classes")
    
    # Test different configurations
    print("\n📊 TESTING DIFFERENT MODEL CONFIGURATIONS")
    print("="*80)
    
    configs = {
        "Quick baseline (current)": {
            'objective': 'multi:softmax',
            'num_class': len(unique_sessions),
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1
        },
        "Deeper trees": {
            'objective': 'multi:softmax',
            'num_class': len(unique_sessions),
            'n_estimators': 100,
            'max_depth': 7,
            'learning_rate': 0.1
        },
        "More trees": {
            'objective': 'multi:softmax',
            'num_class': len(unique_sessions),
            'n_estimators': 500,
            'max_depth': 3,
            'learning_rate': 0.1
        },
        "Deeper + More trees": {
            'objective': 'multi:softmax',
            'num_class': len(unique_sessions),
            'n_estimators': 500,
            'max_depth': 7,
            'learning_rate': 0.1
        },
        "Slow learning": {
            'objective': 'multi:softmax',
            'num_class': len(unique_sessions),
            'n_estimators': 1000,
            'max_depth': 5,
            'learning_rate': 0.01
        }
    }
    
    results = []
    for config_name, params in configs.items():
        print(f"\nTesting: {config_name}")
        result = test_model_config(X, y_numeric, config_name, params)
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.1f}% ± {result['std']:.1f}%")
        print(f"  Training time: {result['training_time']:.2f}s")
    
    # Hyperparameter search
    if len(sys.argv) > 2 and sys.argv[2] == '--full-search':
        search_results = hyperparameter_search(X, y_numeric)
        print(f"\n🏆 HYPERPARAMETER SEARCH RESULTS")
        print(f"Best accuracy: {search_results['best_score']:.1f}%")
        print(f"Best parameters: {search_results['best_params']}")
        print(f"Search time: {search_results['search_time']:.1f}s")
        print(f"Combinations tested: {search_results['n_combinations']}")
    
    # Summary
    print("\n📈 OPTIMIZATION SUMMARY")
    print("="*80)
    print(f"\n{'Configuration':<25} {'Accuracy':<15} {'Time':<10} {'Trees':<10} {'Depth':<10}")
    print("-"*80)
    
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{result['config']:<25} {result['accuracy']:.1f}% ± {result['std']:.1f}%  "
              f"{result['training_time']:.2f}s     "
              f"{result['params'].get('n_estimators', 'N/A'):<10} "
              f"{result['params'].get('max_depth', 'N/A'):<10}")
    
    # Insights
    best_result = max(results, key=lambda x: x['accuracy'])
    baseline_result = next(r for r in results if 'baseline' in r['config'])
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"1. Best configuration: {best_result['config']}")
    print(f"   - Accuracy: {best_result['accuracy']:.1f}%")
    print(f"   - Improvement over baseline: +{best_result['accuracy'] - baseline_result['accuracy']:.1f}%")
    
    print(f"\n2. Training time analysis:")
    print(f"   - Baseline: {baseline_result['training_time']:.2f}s")
    print(f"   - Best model: {best_result['training_time']:.2f}s")
    print(f"   - Time increase: {best_result['training_time']/baseline_result['training_time']:.1f}x")
    
    print(f"\n3. Compute recommendations:")
    print(f"   - Current setup uses single-threaded training")
    print(f"   - Grid search with n_jobs=-1 uses all CPU cores")
    print(f"   - GPU acceleration available with gpu_hist tree method")
    print(f"   - For production: Use early stopping to prevent overfitting")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"   Run full hyperparameter search with: python {sys.argv[0]} {sys.argv[1]} --full-search")
    print(f"   This will test ~144 configurations using all CPU cores")

if __name__ == "__main__":
    main() 