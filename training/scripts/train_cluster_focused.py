"""
Cluster-Focused XGBoost Training

Train XGBoost models using features from specific facial clusters.
Focuses only on differential (_diff) features for movement analysis.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_loader_template import FacemeshDataLoader
from facial_clusters import (
    FACIAL_CLUSTERS, CLUSTER_GROUPS, EXPRESSION_CLUSTERS,
    get_cluster_indices, get_group_indices, 
    get_all_cluster_names, get_all_group_names
)

class ClusterFocusedTrainer:
    """Train models focusing on specific facial clusters"""
    
    def __init__(self, participant_id='e1'):
        self.participant_id = participant_id
        self.results = {}
        
    def get_cluster_features(self, df, cluster_selection, feature_type='diff'):
        """Extract features from specific clusters"""
        feature_cols = []
        
        print(f"  Extracting {feature_type} features from clusters: {cluster_selection}")
        
        for cluster_name in cluster_selection:
            # Check if it's a group or individual cluster
            if cluster_name in CLUSTER_GROUPS:
                # It's a group - get all clusters in the group
                landmark_indices = get_group_indices(cluster_name)
                print(f"    {cluster_name} (group): {len(landmark_indices)} landmarks")
            elif cluster_name in FACIAL_CLUSTERS:
                # It's an individual cluster
                landmark_indices = get_cluster_indices(cluster_name)
                print(f"    {cluster_name} (cluster): {len(landmark_indices)} landmarks")
            else:
                print(f"    Warning: {cluster_name} not found!")
                continue
            
            # Generate feature column names
            for landmark_idx in landmark_indices:
                for axis in ['x', 'y', 'z']:
                    if feature_type == 'diff':
                        col_name = f'feat_{landmark_idx}_{axis}_rb5_diff'
                    elif feature_type == 'rb5':
                        col_name = f'feat_{landmark_idx}_{axis}_rb5'
                    else:  # base
                        col_name = f'feat_{landmark_idx}_{axis}'
                    
                    if col_name in df.columns:
                        feature_cols.append(col_name)
        
        print(f"  Total features extracted: {len(feature_cols)}")
        return feature_cols
    
    def train_cluster_model(self, X, y, target_name, cluster_names, task_type='classification'):
        """Train XGBoost model on cluster-specific features"""
        
        n_samples, n_features = X.shape
        print(f"\n--- CLUSTER-FOCUSED TRAINING: {target_name} ---")
        print(f"  Clusters: {cluster_names}")
        print(f"  Data: {n_samples} samples, {n_features} features")
        print(f"  Target distribution: {y.value_counts().to_dict()}")
        
        # Configure XGBoost based on data size
        if task_type == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                eval_metric='mlogloss'
            )
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Cross-validation
            min_class_count = min(pd.Series(y_encoded).value_counts())
            if min_class_count >= 3:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            else:
                cv = 3
            
            cv_scores = cross_val_score(model, X.values, y_encoded, cv=cv, scoring='accuracy')
            
            # Train final model
            model.fit(X.values, y_encoded)
            
            result = {
                'target_name': target_name,
                'task_type': task_type,
                'clusters_used': cluster_names,
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'n_samples': n_samples,
                'n_features': n_features,
                'n_classes': len(label_encoder.classes_),
                'classes': label_encoder.classes_.tolist(),
                'model': model,
                'label_encoder': label_encoder
            }
            
            print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
        else:  # Regression
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                eval_metric='rmse'
            )
            
            # Cross-validation
            cv_scores_r2 = cross_val_score(model, X.values, y, cv=3, scoring='r2')
            cv_scores_mse = cross_val_score(model, X.values, y, cv=3, scoring='neg_mean_squared_error')
            
            # Train final model
            model.fit(X.values, y)
            
            result = {
                'target_name': target_name,
                'task_type': task_type,
                'clusters_used': cluster_names,
                'cv_r2_mean': cv_scores_r2.mean(),
                'cv_r2_std': cv_scores_r2.std(),
                'cv_mse_mean': -cv_scores_mse.mean(),
                'cv_mse_std': cv_scores_mse.std(),
                'n_samples': n_samples,
                'n_features': n_features,
                'target_range': [float(y.min()), float(y.max())],
                'model': model
            }
            
            print(f"  CV R²: {cv_scores_r2.mean():.3f} ± {cv_scores_r2.std():.3f}")
        
        return result

def load_session_labels():
    """Load session labels"""
    df_labels = pd.read_csv('../../self-reported-data-raw.csv')
    df_labels = df_labels.dropna(subset=['Participant Number', 'Which session number is this for you?\r\n'])
    
    target_columns = {
        'session_type': 'Session Type',
        'stress_level': 'Session experience (1-Low, 5-High)\r\n.Stress level',
        'attention': 'Session experience (1-Low, 5-High)\r\n.Attention',
        'robot_predictability': 'Session experience (1-Low, 5-High)\r\n.Robot movement predictability',
        'excitement': 'Session experience (1-Low, 5-High)\r\n.Excitement',
    }
    
    session_map = {}
    for _, row in df_labels.iterrows():
        participant = str(row['Participant Number']).lower().strip()
        session_num = int(row['Which session number is this for you?\r\n'])
        
        if participant not in session_map:
            session_map[participant] = {}
        
        session_data = {}
        for target_name, col_name in target_columns.items():
            if col_name in df_labels.columns:
                try:
                    value = row[col_name]
                    if target_name == 'session_type':
                        session_data[target_name] = str(value).strip()
                    else:
                        if pd.notna(value) and str(value).strip():
                            session_data[target_name] = float(value)
                        else:
                            session_data[target_name] = None
                except:
                    session_data[target_name] = None
            else:
                session_data[target_name] = None
        
        session_map[participant][session_num] = session_data
    
    return session_map, target_columns

def create_cluster_dataset(participant_id, session_map, cluster_selection, target_name, feature_type='diff'):
    """Create dataset with features from specific clusters"""
    if participant_id not in session_map:
        return None, None, None
    
    participant_sessions = session_map[participant_id]
    loader = FacemeshDataLoader(window_size=5)
    trainer = ClusterFocusedTrainer(participant_id)
    
    all_frames = []
    all_targets = []
    feature_cols = None
    
    for session_num, session_info in participant_sessions.items():
        target_value = session_info.get(target_name)
        if target_value is None:
            continue
        
        df_session = loader.load_subject_data(participant_id, f'session{session_num}')
        if df_session.empty:
            continue
        
        # Extract cluster features (only do this once)
        if feature_cols is None:
            feature_cols = trainer.get_cluster_features(df_session, cluster_selection, feature_type)
        
        # Get features for this session
        session_features = df_session[feature_cols].copy()
        n_frames = len(session_features)
        target_values = [target_value] * n_frames
        
        all_frames.append(session_features)
        all_targets.extend(target_values)
    
    if not all_frames:
        return None, None, None
    
    # Combine all frames
    X = pd.concat(all_frames, ignore_index=True)
    y = pd.Series(all_targets)
    
    # Clean data
    X = X.fillna(0).replace([np.inf, -np.inf], 0).astype(np.float64)
    
    return X, y, feature_cols

def run_cluster_experiments():
    """Run experiments with different cluster combinations"""
    print("=== CLUSTER-FOCUSED FACIAL EXPRESSION TRAINING ===")
    
    # Load data
    session_map, target_columns = load_session_labels()
    participant_id = 'e1'
    
    # Define cluster experiments
    cluster_experiments = {
        'mouth_only': ['mouth'],
        'eyes_only': ['right_eye', 'left_eye'],
        'eyebrows_only': ['eyebrows'],
        'nose_cheeks': ['nose', 'cheeks'],
        'expression_smile': ['mouth', 'cheeks'],  # From EXPRESSION_CLUSTERS
        'expression_surprise': ['mouth', 'eyebrows', 'right_eye', 'left_eye'],
        'expression_concentration': ['eyebrows', 'right_eye', 'left_eye'],
        'all_expression': ['mouth', 'right_eye', 'left_eye', 'eyebrows', 'nose', 'cheeks']
    }
    
    # Targets to test
    targets_to_test = ['session_type', 'robot_predictability', 'attention']
    
    # Initialize trainer
    trainer = ClusterFocusedTrainer(participant_id)
    all_results = {}
    
    print(f"\nAvailable clusters: {get_all_group_names()}")
    print(f"Available individual clusters: {get_all_cluster_names()[:10]}...")  # Show first 10
    
    # Run experiments
    for experiment_name, cluster_selection in cluster_experiments.items():
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {experiment_name.upper()}")
        print(f"{'='*60}")
        
        experiment_results = {}
        
        for target_name in targets_to_test:
            # Create dataset for this cluster combination
            X, y, feature_cols = create_cluster_dataset(
                participant_id, session_map, cluster_selection, target_name, 'diff'
            )
            
            if X is None or len(y.unique()) < 2:
                print(f"Skipping {target_name} - insufficient data")
                continue
            
            # Determine task type
            task_type = 'classification' if target_name == 'session_type' else 'regression'
            
            # Train model
            result = trainer.train_cluster_model(X, y, target_name, cluster_selection, task_type)
            experiment_results[target_name] = result
        
        all_results[experiment_name] = experiment_results
    
    # Print summary
    print_experiment_summary(all_results)
    
    # Save results
    save_cluster_results(all_results, participant_id)
    
    return all_results

def print_experiment_summary(all_results):
    """Print comprehensive summary of all cluster experiments"""
    print(f"\n{'='*80}")
    print("CLUSTER EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    # Create summary table
    summary_data = []
    
    for experiment_name, experiment_results in all_results.items():
        for target_name, result in experiment_results.items():
            row = {
                'Experiment': experiment_name,
                'Target': target_name,
                'Clusters': ' + '.join(result['clusters_used']),
                'Features': result['n_features'],
                'Samples': result['n_samples']
            }
            
            if result['task_type'] == 'classification':
                row['CV_Accuracy'] = f"{result['cv_accuracy_mean']:.3f} ± {result['cv_accuracy_std']:.3f}"
                row['Metric'] = 'Accuracy'
            else:
                row['CV_R2'] = f"{result['cv_r2_mean']:.3f} ± {result['cv_r2_std']:.3f}"
                row['Metric'] = 'R²'
            
            summary_data.append(row)
    
    # Print results grouped by target
    targets = list(set([row['Target'] for row in summary_data]))
    
    for target in targets:
        print(f"\n{target.upper()}:")
        print("-" * 60)
        target_rows = [row for row in summary_data if row['Target'] == target]
        
        for row in target_rows:
            clusters_short = row['Clusters'][:30] + "..." if len(row['Clusters']) > 30 else row['Clusters']
            if 'CV_Accuracy' in row:
                print(f"  {row['Experiment']:20} | {clusters_short:25} | {row['Features']:3}f | {row['CV_Accuracy']}")
            else:
                print(f"  {row['Experiment']:20} | {clusters_short:25} | {row['Features']:3}f | {row['CV_R2']}")
    
    # Find best performers
    print(f"\n{'='*80}")
    print("BEST PERFORMERS")
    print(f"{'='*80}")
    
    for target in targets:
        target_results = [(exp, res[target]) for exp, res in all_results.items() if target in res]
        
        if target_results:
            if target_results[0][1]['task_type'] == 'classification':
                best = max(target_results, key=lambda x: x[1]['cv_accuracy_mean'])
                print(f"{target}: {best[0]} ({best[1]['cv_accuracy_mean']:.3f} accuracy)")
            else:
                best = max(target_results, key=lambda x: x[1]['cv_r2_mean'])
                print(f"{target}: {best[0]} (R² = {best[1]['cv_r2_mean']:.3f})")

def save_cluster_results(all_results, participant_id):
    """Save cluster experiment results"""
    models_dir = Path('models_cluster_focused')
    models_dir.mkdir(exist_ok=True)
    
    # Save full results
    with open(models_dir / f'{participant_id}_cluster_experiments.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Save text summary
    with open(models_dir / f'{participant_id}_cluster_summary.txt', 'w') as f:
        f.write("CLUSTER-FOCUSED TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for experiment_name, experiment_results in all_results.items():
            f.write(f"{experiment_name}:\n")
            for target_name, result in experiment_results.items():
                f.write(f"  {target_name}:\n")
                f.write(f"    Clusters: {result['clusters_used']}\n")
                f.write(f"    Features: {result['n_features']}\n")
                if result['task_type'] == 'classification':
                    f.write(f"    CV Accuracy: {result['cv_accuracy_mean']:.3f} ± {result['cv_accuracy_std']:.3f}\n")
                else:
                    f.write(f"    CV R²: {result['cv_r2_mean']:.3f} ± {result['cv_r2_std']:.3f}\n")
            f.write("\n")
    
    print(f"\nResults saved to {models_dir}/")

def main():
    """Main function"""
    results = run_cluster_experiments()
    return results

if __name__ == "__main__":
    main() 