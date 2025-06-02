"""
Multi-Target Frame Classification Training Script

Train XGBoost models to predict ALL target variables from individual frames
of facial movement data using 3-fold cross-validation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor
import time

# Import our modules
from data_loader_template import FacemeshDataLoader
from facial_clusters import (
    FACIAL_CLUSTERS, CLUSTER_GROUPS, EXPRESSION_CLUSTERS,
    get_cluster_indices, get_group_indices, 
    get_all_cluster_names, get_all_group_names
)

def load_session_labels():
    """Load ALL target variables from self-reported data"""
    # Read the self-reported data
    df_labels = pd.read_csv('../../self-reported-data-raw.csv')
    
    # Clean up the data
    df_labels = df_labels.dropna(subset=['Participant Number', 'Which session number is this for you?\r\n'])
    
    # Define target columns mapping
    target_columns = {
        'session_type': 'Session Type',
        'stress_level': 'Session experience (1-Low, 5-High)\r\n.Stress level',
        'attention': 'Session experience (1-Low, 5-High)\r\n.Attention',
        'robot_predictability': 'Session experience (1-Low, 5-High)\r\n.Robot movement predictability',
        'excitement': 'Session experience (1-Low, 5-High)\r\n.Excitement',
        'mental_demand': 'How mentally demanding was the task?\r\n',
        'rushed_pace': 'How hurried or rushed was the pace of the task?\r\n',
        'work_effort': 'How hard did you have to work to accomplish your level of performance?\r\n',
        'stress_annoyance': 'How insecure, discouraged, irritated, stressed and annoyed were you?\r\n',
        'success_level': 'How successful were you in accomplishing what you needed to do? (1-Failure, 20-Perfect)\r\n'
    }
    
    # Create session mapping with all targets
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
                        # Try to convert to numeric
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

def select_cluster_features(df, feature_clusters, feature_types=['diff']):
    """Select specific feature columns from clusters"""
    feature_cols = []
    
    for cluster in feature_clusters:
        landmark_indices = get_group_indices(cluster) if cluster in CLUSTER_GROUPS else get_cluster_indices(cluster)
        
        for landmark_idx in landmark_indices:
            for axis in ['x', 'y', 'z']:
                for ftype in feature_types:
                    if ftype == 'base':
                        col_name = f'feat_{landmark_idx}_{axis}'
                    elif ftype == 'rb':
                        col_name = f'feat_{landmark_idx}_{axis}_rb5'
                    elif ftype == 'diff':
                        col_name = f'feat_{landmark_idx}_{axis}_rb5_diff'
                    
                    if col_name in df.columns:
                        feature_cols.append(col_name)
    
    return feature_cols

def create_frame_dataset(participant_id, session_map, feature_clusters, target_name):
    """Create dataset where each frame is a sample for a specific target"""
    if participant_id not in session_map:
        return None, None, None
    
    participant_sessions = session_map[participant_id]
    
    # Load data for all sessions
    loader = FacemeshDataLoader(window_size=5)
    all_frames = []
    all_targets = []
    
    for session_num, session_info in participant_sessions.items():
        target_value = session_info.get(target_name)
        
        if target_value is None:
            continue  # Skip sessions without this target
        
        # Load session data
        df_session = loader.load_subject_data(participant_id, f'session{session_num}')
        
        if df_session.empty:
            continue
        
        # Select feature columns (same for all sessions)
        if not all_frames:  # First session - determine feature columns
            feature_cols = select_cluster_features(df_session, feature_clusters)
        
        # Add each frame as a separate sample
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
    
    # Handle missing values and ensure numeric types
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Convert to float64 to ensure compatibility with XGBoost
    X = X.astype(np.float64)
    
    return X, y, feature_cols

def train_single_target_model(participant_id, session_map, feature_clusters, target_name, target_type='classification'):
    """Train model for a single target variable"""
    print(f"\n--- Training {target_name} model ---")
    
    # Create dataset
    X, y, feature_cols = create_frame_dataset(participant_id, session_map, feature_clusters, target_name)
    
    if X is None or len(y.unique()) < 2:
        print(f"  Insufficient data for {target_name}")
        return None
    
    print(f"  Dataset: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
    
    # Determine if classification or regression
    if target_type == 'classification' or target_name == 'session_type':
        # Classification
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Check if we have enough samples for stratification
        min_class_count = min(pd.Series(y_encoded).value_counts())
        if min_class_count >= 3:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        else:
            cv = 3
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X.values, y_encoded, cv=cv, scoring='accuracy')
        
        # Train final model on all data
        model.fit(X.values, y_encoded)
        
        result = {
            'target_name': target_name,
            'target_type': 'classification',
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(label_encoder.classes_),
            'classes': label_encoder.classes_.tolist(),
            'model': model,
            'label_encoder': label_encoder,
            'feature_columns': feature_cols
        }
        
    else:
        # Regression
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Cross-validation for regression
        cv_scores_r2 = cross_val_score(model, X.values, y, cv=3, scoring='r2')
        cv_scores_mse = cross_val_score(model, X.values, y, cv=3, scoring='neg_mean_squared_error')
        
        # Train final model on all data
        model.fit(X.values, y)
        
        result = {
            'target_name': target_name,
            'target_type': 'regression',
            'cv_r2_mean': cv_scores_r2.mean(),
            'cv_r2_std': cv_scores_r2.std(),
            'cv_mse_mean': -cv_scores_mse.mean(),
            'cv_mse_std': cv_scores_mse.std(),
            'cv_r2_scores': cv_scores_r2.tolist(),
            'cv_mse_scores': (-cv_scores_mse).tolist(),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'target_range': [float(y.min()), float(y.max())],
            'model': model,
            'feature_columns': feature_cols
        }
    
    print(f"  {target_name} - CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})" if target_type == 'classification' else f"  {target_name} - CV R²: {cv_scores_r2.mean():.3f} (+/- {cv_scores_r2.std():.3f})")
    
    return result

def train_all_models(participant_id, session_map, feature_clusters, target_columns):
    """Train models for all target variables"""
    print(f"\n{'='*60}")
    print(f"TRAINING ALL MODELS FOR {participant_id.upper()}")
    print(f"{'='*60}")
    
    # Define which targets are classification vs regression
    classification_targets = ['session_type']
    regression_targets = [name for name in target_columns.keys() if name not in classification_targets]
    
    all_results = {}
    
    # Train classification models
    for target_name in classification_targets:
        result = train_single_target_model(participant_id, session_map, feature_clusters, target_name, 'classification')
        if result:
            all_results[target_name] = result
    
    # Train regression models
    for target_name in regression_targets:
        result = train_single_target_model(participant_id, session_map, feature_clusters, target_name, 'regression')
        if result:
            all_results[target_name] = result
    
    return all_results

def save_all_models(participant_id, all_results):
    """Save all trained models"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    for target_name, result in all_results.items():
        model_path = models_dir / f'{participant_id}_{target_name}_model.pkl'
        
        # Save only the necessary parts (not the large feature columns)
        save_data = {
            'model': result['model'],
            'target_name': result['target_name'],
            'target_type': result['target_type'],
            'feature_columns': result['feature_columns']
        }
        
        if 'label_encoder' in result:
            save_data['label_encoder'] = result['label_encoder']
        
        with open(model_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved {target_name} model to {model_path}")

def print_summary_results(all_results):
    """Print comprehensive summary of all model results"""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    classification_results = []
    regression_results = []
    
    for target_name, result in all_results.items():
        if result['target_type'] == 'classification':
            classification_results.append({
                'Target': target_name,
                'CV Accuracy': f"{result['cv_accuracy_mean']:.3f} ± {result['cv_accuracy_std']:.3f}",
                'Samples': result['n_samples'],
                'Features': result['n_features'],
                'Classes': result['n_classes']
            })
        else:
            regression_results.append({
                'Target': target_name,
                'CV R²': f"{result['cv_r2_mean']:.3f} ± {result['cv_r2_std']:.3f}",
                'CV MSE': f"{result['cv_mse_mean']:.3f} ± {result['cv_mse_std']:.3f}",
                'Samples': result['n_samples'],
                'Features': result['n_features'],
                'Range': f"[{result['target_range'][0]:.1f}, {result['target_range'][1]:.1f}]"
            })
    
    if classification_results:
        print("\nCLASSIFICATION MODELS:")
        print("-" * 80)
        df_class = pd.DataFrame(classification_results)
        print(df_class.to_string(index=False))
    
    if regression_results:
        print("\nREGRESSION MODELS:")
        print("-" * 80)
        df_reg = pd.DataFrame(regression_results)
        print(df_reg.to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"TOTAL MODELS TRAINED: {len(all_results)}")
    
    # Best performing models
    if classification_results:
        best_class = max(all_results.items(), 
                        key=lambda x: x[1]['cv_accuracy_mean'] if x[1]['target_type'] == 'classification' else 0)
        print(f"BEST CLASSIFICATION: {best_class[0]} ({best_class[1]['cv_accuracy_mean']:.3f} accuracy)")
    
    if regression_results:
        best_reg = max(all_results.items(), 
                      key=lambda x: x[1]['cv_r2_mean'] if x[1]['target_type'] == 'regression' else -999)
        print(f"BEST REGRESSION: {best_reg[0]} (R² = {best_reg[1]['cv_r2_mean']:.3f})")
    
    print(f"{'='*80}")

def main():
    """Main training function"""
    start_time = time.time()
    
    print("=== MULTI-TARGET FACIAL EXPRESSION CLASSIFICATION ===")
    
    # Load session labels and target definitions
    print("Loading session labels and target definitions...")
    session_map, target_columns = load_session_labels()
    print(f"Loaded labels for {len(session_map)} participants")
    print(f"Target variables: {list(target_columns.keys())}")
    
    # Train models for e1
    participant_id = 'e1'
    
    # Use all expression-related clusters plus movement features
    feature_clusters = ['mouth', 'right_eye', 'left_eye', 'eyebrows', 'nose', 'cheeks']
    feature_types = ['diff']  # Focus on movement differences
    
    print(f"\nUsing feature clusters: {feature_clusters}")
    print(f"Using feature types: {feature_types}")
    
    # Train all models
    all_results = train_all_models(participant_id, session_map, feature_clusters, target_columns)
    
    # Save all models
    if all_results:
        print(f"\n{'='*60}")
        print("SAVING ALL MODELS")
        print(f"{'='*60}")
        save_all_models(participant_id, all_results)
        
        # Print comprehensive summary
        print_summary_results(all_results)
        
        # Save summary to file
        summary_path = f'models/{participant_id}_model_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("MODEL TRAINING SUMMARY\n")
            f.write("="*50 + "\n\n")
            for target_name, result in all_results.items():
                f.write(f"{target_name}:\n")
                if result['target_type'] == 'classification':
                    f.write(f"  CV Accuracy: {result['cv_accuracy_mean']:.3f} ± {result['cv_accuracy_std']:.3f}\n")
                    f.write(f"  Classes: {result['n_classes']}\n")
                else:
                    f.write(f"  CV R²: {result['cv_r2_mean']:.3f} ± {result['cv_r2_std']:.3f}\n")
                    f.write(f"  CV MSE: {result['cv_mse_mean']:.3f} ± {result['cv_mse_std']:.3f}\n")
                f.write(f"  Samples: {result['n_samples']}\n")
                f.write(f"  Features: {result['n_features']}\n\n")
        
        print(f"\nSummary saved to {summary_path}")
    
    end_time = time.time()
    print(f"\nTotal training time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 