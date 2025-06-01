"""
Frame-based Session Classification Training Script

Train XGBoost models to predict session types from individual frames
of facial movement data. Each frame is a separate training sample.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
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
    
    # Create session mapping (including stress levels)
    session_map = {}
    for _, row in df_labels.iterrows():
        participant = str(row['Participant Number']).lower().strip()
        session_num = int(row['Which session number is this for you?\r\n'])
        session_type = str(row['Session Type']).strip()
        
        # Extract stress level if available
        stress_col = 'Session experience (1-Low, 5-High)\r\n.Stress level'
        stress_level = None
        if stress_col in df_labels.columns:
            try:
                stress_level = int(row[stress_col])
            except:
                stress_level = None
        
        if participant not in session_map:
            session_map[participant] = {}
        session_map[participant][session_num] = {
            'session_type': session_type,
            'stress_level': stress_level
        }
    
    return session_map

def select_cluster_features(df, feature_clusters, feature_types=['base', 'rb', 'diff']):
    """
    Select specific feature columns from clusters
    
    Args:
        df: DataFrame with facial data
        feature_clusters: List of clusters to use
        feature_types: Types of features to include
    
    Returns:
        List of column names to use as features
    """
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

def create_frame_dataset(participant_id, session_map, feature_clusters, target_type='session'):
    """
    Create dataset where each frame is a sample
    
    Args:
        participant_id: ID of participant (e.g., 'e1')
        session_map: Dictionary mapping participants to session info
        feature_clusters: List of facial clusters to use
        target_type: 'session' for session type, 'stress' for stress level
    
    Returns:
        X (features), y (targets), feature_names
    """
    print(f"\n=== Creating frame dataset for {participant_id} ===")
    
    if participant_id not in session_map:
        print(f"No session labels found for {participant_id}")
        return None, None, None
    
    participant_sessions = session_map[participant_id]
    print(f"Found {len(participant_sessions)} sessions")
    
    # Load data for all sessions
    loader = FacemeshDataLoader(window_size=5)
    all_frames = []
    all_targets = []
    
    for session_num, session_info in participant_sessions.items():
        session_type = session_info['session_type']
        stress_level = session_info['stress_level']
        
        print(f"Loading session {session_num} (type {session_type}, stress {stress_level})...")
        
        # Load session data
        df_session = loader.load_subject_data(participant_id, f'session{session_num}')
        
        if df_session.empty:
            print(f"  Warning: No data found for session {session_num}")
            continue
        
        # Select feature columns (same for all sessions)
        if not all_frames:  # First session - determine feature columns
            feature_cols = select_cluster_features(df_session, feature_clusters)
            print(f"Selected {len(feature_cols)} feature columns from {len(feature_clusters)} clusters")
        
        # Add each frame as a separate sample
        session_features = df_session[feature_cols].copy()
        n_frames = len(session_features)
        
        # Choose target based on target_type
        if target_type == 'stress' and stress_level is not None:
            target_values = [stress_level] * n_frames
        else:
            target_values = [session_type] * n_frames
        
        all_frames.append(session_features)
        all_targets.extend(target_values)
        print(f"  Added {n_frames} frames")
    
    if not all_frames:
        print("No data loaded")
        return None, None, None
    
    # Combine all frames
    X = pd.concat(all_frames, ignore_index=True)
    y = pd.Series(all_targets)
    
    # Handle missing values
    X = X.fillna(0)
    
    # Replace inf values
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"\nDataset created:")
    print(f"  Shape: {X.shape}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols

def train_frame_model(participant_id, session_map, feature_clusters, target_type='session'):
    """
    Train XGBoost model using individual frames
    
    Args:
        participant_id: ID of participant (e.g., 'e1')
        session_map: Dictionary mapping participants to session info
        feature_clusters: List of facial clusters to use
        target_type: 'session' for session type, 'stress' for stress level
    
    Returns:
        Trained model and performance metrics
    """
    # Create dataset
    X, y, feature_cols = create_frame_dataset(participant_id, session_map, feature_clusters, target_type)
    
    if X is None:
        return None, None
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTrain/Test split:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    print(cm)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    performance = {
        'accuracy': accuracy,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_train': len(X_train),
        'n_test': len(X_test),
        'target_distribution': y.value_counts().to_dict(),
        'confusion_matrix': cm.tolist()
    }
    
    return {
        'model': model,
        'label_encoder': label_encoder,
        'feature_columns': feature_cols,
        'performance': performance,
        'feature_importance': feature_importance,
        'target_type': target_type
    }, performance

def main():
    """Main training function"""
    print("=== Frame-based Classification Training ===")
    
    # Load session labels
    print("Loading session labels...")
    session_map = load_session_labels()
    print(f"Loaded labels for {len(session_map)} participants")
    
    # Train model for e1
    participant_id = 'e1'
    
    # Use expression-related clusters + movement features
    feature_clusters = ['mouth', 'right_eye', 'left_eye', 'eyebrows', 'nose', 'cheeks']
    
    # Train session type classifier
    print("\n" + "="*50)
    print("TRAINING SESSION TYPE CLASSIFIER")
    print("="*50)
    
    model_info_session, performance_session = train_frame_model(
        participant_id, 
        session_map, 
        feature_clusters,
        target_type='session'
    )
    
    if model_info_session:
        # Save model
        model_path = f'models/{participant_id}_frame_session_classifier.pkl'
        Path('models').mkdir(exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_info_session, f)
        
        print(f"\nSession classifier saved to {model_path}")
    
    # Train stress level classifier
    print("\n" + "="*50)
    print("TRAINING STRESS LEVEL CLASSIFIER")
    print("="*50)
    
    model_info_stress, performance_stress = train_frame_model(
        participant_id, 
        session_map, 
        feature_clusters,
        target_type='stress'
    )
    
    if model_info_stress:
        # Save model
        model_path = f'models/{participant_id}_frame_stress_classifier.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_info_stress, f)
        
        print(f"\nStress classifier saved to {model_path}")
    
    # Summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    if performance_session:
        print(f"Session Type Accuracy: {performance_session['accuracy']:.3f}")
    if performance_stress:
        print(f"Stress Level Accuracy: {performance_stress['accuracy']:.3f}")

if __name__ == "__main__":
    main() 