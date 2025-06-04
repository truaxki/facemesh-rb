"""
Session Classification Training Script

Train XGBoost models to predict session types for individual participants
based on facial movement features from clusters.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
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

def load_session_labels():
    """Load session type labels from self-reported data"""
    # Read the self-reported data
    df_labels = pd.read_csv('../../self-reported-data-raw.csv')
    
    # Clean up the data
    df_labels = df_labels.dropna(subset=['Participant Number', 'Which session number is this for you?\r\n'])
    
    # Create session mapping
    session_map = {}
    for _, row in df_labels.iterrows():
        participant = str(row['Participant Number']).lower().strip()
        session_num = int(row['Which session number is this for you?\r\n'])
        session_type = str(row['Session Type']).strip()
        
        if participant not in session_map:
            session_map[participant] = {}
        session_map[participant][session_num] = session_type
    
    return session_map

def extract_session_features(df, feature_clusters=None, feature_types=['base', 'rb', 'diff']):
    """
    Extract aggregated features per session from facial data
    
    Args:
        df: DataFrame with facial data
        feature_clusters: List of clusters to use (default: all expression clusters)
        feature_types: Types of features to include
    
    Returns:
        Dictionary with aggregated features
    """
    if feature_clusters is None:
        # Use expression-related clusters by default
        feature_clusters = []
        for expr_groups in EXPRESSION_CLUSTERS.values():
            feature_clusters.extend(expr_groups)
        feature_clusters = list(set(feature_clusters))  # Remove duplicates
    
    # Collect all feature columns
    feature_cols = []
    loader = FacemeshDataLoader(window_size=5)  # For rb5 features
    
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
    
    print(f"Selected {len(feature_cols)} feature columns from {len(feature_clusters)} clusters")
    
    if not feature_cols:
        print("Warning: No feature columns found!")
        return {}
    
    # Calculate aggregated statistics
    feature_data = df[feature_cols]
    
    features = {}
    
    # Basic statistics - ensure numeric types and handle NaN/inf
    features.update({f'{col}_mean': float(feature_data[col].mean()) for col in feature_cols})
    features.update({f'{col}_std': float(feature_data[col].std()) for col in feature_cols})
    features.update({f'{col}_min': float(feature_data[col].min()) for col in feature_cols})
    features.update({f'{col}_max': float(feature_data[col].max()) for col in feature_cols})
    features.update({f'{col}_range': float(feature_data[col].max() - feature_data[col].min()) for col in feature_cols})
    
    # Movement-specific features (for diff features)
    diff_cols = [col for col in feature_cols if 'diff' in col]
    if diff_cols:
        diff_data = feature_data[diff_cols]
        # Overall movement magnitude
        movement_magnitude = np.sqrt(np.sum(diff_data**2, axis=1))
        features['movement_magnitude_mean'] = float(movement_magnitude.mean())
        features['movement_magnitude_std'] = float(movement_magnitude.std())
        features['movement_magnitude_max'] = float(movement_magnitude.max())
        features['movement_peaks'] = int(len([x for x in movement_magnitude if x > movement_magnitude.mean() + 2*movement_magnitude.std()]))
    
    # Replace any NaN or inf values with 0
    for key, value in features.items():
        if pd.isna(value) or np.isinf(value):
            features[key] = 0.0
    
    return features

def train_participant_model(participant_id, session_map, feature_clusters=None):
    """
    Train XGBoost model for a specific participant
    
    Args:
        participant_id: ID of participant (e.g., 'e1')
        session_map: Dictionary mapping participants to session types
        feature_clusters: List of facial clusters to use
    
    Returns:
        Trained model and performance metrics
    """
    print(f"\n=== Training model for {participant_id} ===")
    
    # Check if participant has session labels
    if participant_id not in session_map:
        print(f"No session labels found for {participant_id}")
        return None, None
    
    participant_sessions = session_map[participant_id]
    print(f"Found {len(participant_sessions)} sessions: {participant_sessions}")
    
    # Load data for all sessions
    loader = FacemeshDataLoader(window_size=5)
    all_features = []
    all_labels = []
    
    for session_num, session_type in participant_sessions.items():
        print(f"Loading session {session_num} (type {session_type})...")
        
        # Load session data
        df_session = loader.load_subject_data(participant_id, f'session{session_num}')
        
        if df_session.empty:
            print(f"  Warning: No data found for session {session_num}")
            continue
        
        # Extract features for this session
        session_features = extract_session_features(df_session, feature_clusters)
        
        if session_features:
            all_features.append(session_features)
            all_labels.append(session_type)
            print(f"  Extracted {len(session_features)} features")
        else:
            print(f"  Warning: No features extracted for session {session_num}")
    
    if len(all_features) < 2:
        print(f"Insufficient data for training (only {len(all_features)} sessions with features)")
        return None, None
    
    # Convert to DataFrame
    X = pd.DataFrame(all_features)
    y = pd.Series(all_labels)
    
    print(f"\nTraining data shape: {X.shape}")
    print(f"Session types: {y.value_counts().to_dict()}")
    
    # Handle missing values
    X = X.fillna(0)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # With only 8 sessions, use cross-validation instead of train/test split
    # since each session type appears only once
    print("Using cross-validation (each session type appears only once)")
    
    # Train on all data
    model.fit(X, y_encoded)
    
    # Use leave-one-out cross-validation
    loo = LeaveOneOut()
    
    cv_predictions = []
    cv_true = []
    
    for train_idx, test_idx in loo.split(X):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y_encoded[train_idx], y_encoded[test_idx]
        
        # Train model on n-1 samples
        model_cv = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        model_cv.fit(X_train_cv, y_train_cv)
        
        # Predict on the held-out sample
        pred = model_cv.predict(X_test_cv)
        cv_predictions.extend(pred)
        cv_true.extend(y_test_cv)
    
    # Calculate accuracy
    cv_accuracy = accuracy_score(cv_true, cv_predictions)
    
    # Convert predictions back to labels for reporting
    cv_pred_labels = label_encoder.inverse_transform(cv_predictions)
    cv_true_labels = label_encoder.inverse_transform(cv_true)
    
    print(f"\nLeave-One-Out Cross-Validation Accuracy: {cv_accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(cv_true_labels, cv_pred_labels))
    
    performance = {
        'loo_accuracy': cv_accuracy,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'sessions': list(y.unique()),
        'confusion_matrix': confusion_matrix(cv_true_labels, cv_pred_labels).tolist()
    }
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    return {
        'model': model,
        'label_encoder': label_encoder,
        'feature_columns': X.columns.tolist(),
        'performance': performance,
        'feature_importance': feature_importance
    }, performance

def main():
    """Main training function"""
    print("=== Session Classification Training ===")
    
    # Load session labels
    print("Loading session labels...")
    session_map = load_session_labels()
    print(f"Loaded labels for {len(session_map)} participants")
    
    # Train model for e1
    participant_id = 'e1'
    
    # Use expression-related clusters
    feature_clusters = ['mouth', 'right_eye', 'left_eye', 'eyebrows', 'nose', 'cheeks']
    
    model_info, performance = train_participant_model(
        participant_id, 
        session_map, 
        feature_clusters
    )
    
    if model_info:
        # Save model
        model_path = f'models/{participant_id}_session_classifier.pkl'
        Path('models').mkdir(exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"\nModel saved to {model_path}")
        print(f"Final performance: {performance}")
    else:
        print(f"Failed to train model for {participant_id}")

if __name__ == "__main__":
    main() 