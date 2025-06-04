"""
Optimized Multi-Target Training with Feature Selection and Regularization

Addresses high-dimensional data challenges with feature selection,
regularization, and unsupervised pre-training options.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
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

class OptimizedFacemeshTrainer:
    """Optimized trainer for high-dimensional facial expression data"""
    
    def __init__(self, participant_id='e1'):
        self.participant_id = participant_id
        self.models = {}
        self.feature_selectors = {}
        self.scalers = {}
        
    def get_optimized_xgb_params(self, n_samples, n_features, task_type='classification'):
        """Get optimized XGBoost parameters based on data dimensions"""
        
        # Adjust based on sample-to-feature ratio
        sample_feature_ratio = n_samples / n_features
        
        if sample_feature_ratio < 2:  # High-dimensional case
            params = {
                'n_estimators': 50,          # Fewer trees to prevent overfitting
                'max_depth': 3,              # Shallow trees
                'learning_rate': 0.05,       # Slower learning
                'subsample': 0.8,            # Sample 80% of data
                'colsample_bytree': 0.6,     # Use 60% of features per tree
                'reg_alpha': 1.0,            # L1 regularization
                'reg_lambda': 1.0,           # L2 regularization
                'random_state': 42
            }
        elif sample_feature_ratio < 5:  # Medium-dimensional
            params = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'subsample': 0.9,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'random_state': 42
            }
        else:  # Low-dimensional (plenty of samples)
            params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42
            }
        
        # Add task-specific parameters
        if task_type == 'classification':
            params['eval_metric'] = 'mlogloss'
            return xgb.XGBClassifier(**params)
        else:
            params['eval_metric'] = 'rmse'
            return xgb.XGBRegressor(**params)
    
    def select_features(self, X, y, task_type='classification', n_features=100):
        """Intelligent feature selection"""
        print(f"  Selecting top {n_features} features from {X.shape[1]}...")
        
        if task_type == 'classification':
            # Use mutual information for classification
            selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, X.shape[1]))
        else:
            # Use F-test for regression
            selector = SelectKBest(score_func=f_regression, k=min(n_features, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = np.array(X.columns)[selector.get_support()]
        
        print(f"  Selected {X_selected.shape[1]} features")
        return X_selected, selector, selected_features
    
    def unsupervised_exploration(self, X, target_name, n_clusters=8):
        """Unsupervised analysis to understand data structure"""
        print(f"\n=== UNSUPERVISED ANALYSIS: {target_name} ===")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA Analysis
        pca = PCA(n_components=min(50, X.shape[1], X.shape[0]-1))
        X_pca = pca.fit_transform(X_scaled)
        
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        print(f"  PCA: {len(pca.components_)} components explain {explained_variance[-1]:.3f} variance")
        print(f"  First 10 components explain {explained_variance[9]:.3f} variance")
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_pca[:, :10])  # Use first 10 PCA components
        
        silhouette = silhouette_score(X_pca[:, :10], clusters)
        print(f"  K-Means ({n_clusters} clusters): Silhouette score = {silhouette:.3f}")
        
        # Analyze cluster distribution
        cluster_counts = np.bincount(clusters)
        print(f"  Cluster sizes: {cluster_counts}")
        
        return {
            'pca': pca,
            'scaler': scaler,
            'clusters': clusters,
            'silhouette_score': silhouette,
            'explained_variance': explained_variance,
            'X_pca': X_pca
        }
    
    def train_with_feature_selection(self, X, y, target_name, task_type='classification'):
        """Train model with automatic feature selection and optimization"""
        print(f"\n--- OPTIMIZED TRAINING: {target_name} ---")
        
        n_samples, n_features = X.shape
        sample_feature_ratio = n_samples / n_features
        
        print(f"  Data: {n_samples} samples, {n_features} features")
        print(f"  Sample/Feature ratio: {sample_feature_ratio:.2f}")
        
        # Determine optimal number of features (rule of thumb: 10 samples per feature)
        optimal_features = min(n_features, max(10, n_samples // 10))
        print(f"  Target features: {optimal_features}")
        
        # Feature selection
        X_selected, selector, selected_features = self.select_features(
            X, y, task_type, optimal_features
        )
        
        # Unsupervised exploration
        unsupervised_results = self.unsupervised_exploration(X_selected, target_name)
        
        # Get optimized model
        model = self.get_optimized_xgb_params(n_samples, X_selected.shape[1], task_type)
        
        # Cross-validation
        if task_type == 'classification':
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Stratified CV if enough samples per class
            min_class_count = min(pd.Series(y_encoded).value_counts())
            if min_class_count >= 3:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            else:
                cv = 3
            
            cv_scores = cross_val_score(model, X_selected, y_encoded, cv=cv, scoring='accuracy')
            model.fit(X_selected, y_encoded)
            
            result = {
                'target_name': target_name,
                'task_type': task_type,
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'n_samples': n_samples,
                'n_features_original': n_features,
                'n_features_selected': X_selected.shape[1],
                'selected_features': selected_features.tolist(),
                'sample_feature_ratio': sample_feature_ratio,
                'model': model,
                'feature_selector': selector,
                'label_encoder': label_encoder,
                'unsupervised_results': unsupervised_results
            }
            
        else:  # Regression
            cv_scores_r2 = cross_val_score(model, X_selected, y, cv=3, scoring='r2')
            cv_scores_mse = cross_val_score(model, X_selected, y, cv=3, scoring='neg_mean_squared_error')
            model.fit(X_selected, y)
            
            result = {
                'target_name': target_name,
                'task_type': task_type,
                'cv_r2_mean': cv_scores_r2.mean(),
                'cv_r2_std': cv_scores_r2.std(),
                'cv_mse_mean': -cv_scores_mse.mean(),
                'cv_mse_std': cv_scores_mse.std(),
                'n_samples': n_samples,
                'n_features_original': n_features,
                'n_features_selected': X_selected.shape[1],
                'selected_features': selected_features.tolist(),
                'sample_feature_ratio': sample_feature_ratio,
                'model': model,
                'feature_selector': selector,
                'unsupervised_results': unsupervised_results
            }
        
        print(f"  Features reduced: {n_features} → {X_selected.shape[1]}")
        print(f"  Unsupervised clusters: Silhouette = {unsupervised_results['silhouette_score']:.3f}")
        
        if task_type == 'classification':
            print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        else:
            print(f"  CV R²: {cv_scores_r2.mean():.3f} ± {cv_scores_r2.std():.3f}")
        
        return result

def load_session_labels():
    """Load session labels (reusing from original script)"""
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

def create_frame_dataset(participant_id, session_map, feature_clusters, target_name):
    """Create frame-level dataset (reusing from original)"""
    if participant_id not in session_map:
        return None, None, None
    
    participant_sessions = session_map[participant_id]
    loader = FacemeshDataLoader(window_size=5)
    all_frames = []
    all_targets = []
    
    def select_cluster_features(df, feature_clusters, feature_types=['diff']):
        feature_cols = []
        for cluster in feature_clusters:
            landmark_indices = get_group_indices(cluster) if cluster in CLUSTER_GROUPS else get_cluster_indices(cluster)
            for landmark_idx in landmark_indices:
                for axis in ['x', 'y', 'z']:
                    for ftype in feature_types:
                        if ftype == 'diff':
                            col_name = f'feat_{landmark_idx}_{axis}_rb5_diff'
                        if col_name in df.columns:
                            feature_cols.append(col_name)
        return feature_cols
    
    for session_num, session_info in participant_sessions.items():
        target_value = session_info.get(target_name)
        if target_value is None:
            continue
        
        df_session = loader.load_subject_data(participant_id, f'session{session_num}')
        if df_session.empty:
            continue
        
        if not all_frames:
            feature_cols = select_cluster_features(df_session, feature_clusters)
        
        session_features = df_session[feature_cols].copy()
        n_frames = len(session_features)
        target_values = [target_value] * n_frames
        
        all_frames.append(session_features)
        all_targets.extend(target_values)
    
    if not all_frames:
        return None, None, None
    
    X = pd.concat(all_frames, ignore_index=True)
    y = pd.Series(all_targets)
    
    # Clean data
    X = X.fillna(0).replace([np.inf, -np.inf], 0).astype(np.float64)
    
    return X, y, feature_cols

def main():
    """Main optimized training function"""
    print("=== OPTIMIZED MULTI-TARGET TRAINING ===")
    
    # Load data
    session_map, target_columns = load_session_labels()
    participant_id = 'e1'
    feature_clusters = ['mouth', 'right_eye', 'left_eye', 'eyebrows', 'nose', 'cheeks']
    
    # Initialize trainer
    trainer = OptimizedFacemeshTrainer(participant_id)
    
    # Train a few key models with optimization
    key_targets = ['session_type', 'stress_level', 'attention', 'robot_predictability']
    
    results = {}
    
    for target_name in key_targets:
        # Load data
        X, y, feature_cols = create_frame_dataset(participant_id, session_map, feature_clusters, target_name)
        
        if X is None or len(y.unique()) < 2:
            print(f"Skipping {target_name} - insufficient data")
            continue
        
        # Determine task type
        task_type = 'classification' if target_name == 'session_type' else 'regression'
        
        # Train with optimization
        result = trainer.train_with_feature_selection(X, y, target_name, task_type)
        results[target_name] = result
    
    # Print summary
    print(f"\n{'='*80}")
    print("OPTIMIZED TRAINING SUMMARY")
    print(f"{'='*80}")
    
    for target_name, result in results.items():
        print(f"\n{target_name}:")
        print(f"  Features: {result['n_features_original']} → {result['n_features_selected']}")
        print(f"  Sample/Feature ratio: {result['sample_feature_ratio']:.2f}")
        print(f"  Silhouette score: {result['unsupervised_results']['silhouette_score']:.3f}")
        
        if result['task_type'] == 'classification':
            print(f"  CV Accuracy: {result['cv_accuracy_mean']:.3f} ± {result['cv_accuracy_std']:.3f}")
        else:
            print(f"  CV R²: {result['cv_r2_mean']:.3f} ± {result['cv_r2_std']:.3f}")
    
    # Save results
    models_dir = Path('models_optimized')
    models_dir.mkdir(exist_ok=True)
    
    with open(models_dir / f'{participant_id}_optimized_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nOptimized results saved to {models_dir}/")

if __name__ == "__main__":
    main() 