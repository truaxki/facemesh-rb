"""
Focused Kabsch Alignment for SVM Training
=========================================

Processes only Magic 6 landmarks (nose + cheeks) and creates a compact dataset with:
- Raw coordinates (current frame)
- Rolling baseline coordinates (5-frame average) 
- Kabsch-transformed coordinates (current aligned to baseline)

Total features: 6 landmarks × 3 coordinate sets × 3 axes = 54 features

Author: Facial Expression Analysis System
Date: December 2024
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report
from scipy.linalg import svd, det
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Self-contained Kabsch algorithm implementation
class KabschAlgorithm:
    """Self-contained Kabsch algorithm implementation."""
    
    @staticmethod
    def kabsch_algorithm(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Kabsch algorithm for optimal rotation matrix calculation.
        
        Args:
            P: Reference point set (N x 3)
            Q: Point set to align to P (N x 3)
            
        Returns:
            R: Optimal rotation matrix (3 x 3)
            t: Translation vector (3,)
            rmsd: Root mean square deviation after alignment
        """
        assert P.shape == Q.shape, "Point sets must have same shape"
        assert P.shape[1] == 3, "Points must be 3D"
        
        # Step 1: Center both point sets (translation)
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        
        # Step 2: Compute cross-covariance matrix H
        H = P_centered.T @ Q_centered
        
        # Step 3: Singular Value Decomposition
        U, S, Vt = svd(H)
        
        # Step 4: Compute rotation matrix
        # Check for reflection case
        d = det(U @ Vt)
        
        if d < 0:
            # Reflection case - flip the last column of U
            U[:, -1] *= -1
        
        R = U @ Vt
        
        # Step 5: Compute translation
        t = centroid_P - R @ centroid_Q
        
        # Step 6: Calculate RMSD
        Q_aligned = (R @ Q_centered.T).T + centroid_P
        rmsd = np.sqrt(np.mean(np.sum((P - Q_aligned)**2, axis=1)))
        
        return R, t, rmsd
    
    @staticmethod
    def apply_transformation(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Apply rotation and translation to point cloud.
        
        Args:
            points: Point cloud (N x 3)
            R: Rotation matrix (3 x 3)
            t: Translation vector (3,)
            
        Returns:
            Transformed points (N x 3)
        """
        return (R @ points.T).T + t

class FocusedKabschProcessor:
    """
    Focused Kabsch processor: Magic 6 for rotation, expanded set for features.
    """
    
    def __init__(self):
        # Magic 6 stable landmarks for rotation computation: nose + cheeks
        self.rotation_landmarks = [1, 2, 98, 327, 205, 425]
        self.rotation_names = ['noseTip', 'noseBottom', 'noseRightCorner', 
                              'noseLeftCorner', 'rightCheek', 'leftCheek']
        
        # Expanded landmark set for feature extraction: nose + cheeks + eyes + lips
        self.feature_landmarks = self._get_expanded_landmarks()
        self.feature_names = self._get_expanded_names()
        
        print(f"🎯 Rotation landmarks: {len(self.rotation_landmarks)} (Magic 6)")
        print(f"📊 Feature landmarks: {len(self.feature_landmarks)} (nose + cheeks + eyes + lips)")
    
    def _get_expanded_landmarks(self):
        """Get expanded landmark set including nose, cheeks, eyes, and lips."""
        # Start with Magic 6 (nose + cheeks)
        landmarks = set([1, 2, 98, 327, 205, 425])
        
        # Add key lip landmarks (simplified selection)
        lips = [
            # Upper lip outer
            161, 185, 40, 39, 37, 0, 267, 269, 270, 409, 29,
            # Lower lip outer  
            146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            # Key inner lip points
            78, 13, 308, 14, 317, 402
        ]
        landmarks.update(lips)
        
        # Add key eye landmarks (simplified selection)
        eyes = [
            # Right eye key points
            246, 161, 160, 159, 158, 157, 173,  # Upper
            33, 7, 163, 144, 145, 153, 154, 155, 133,  # Lower
            # Left eye key points
            466, 388, 387, 386, 385, 384, 398,  # Upper
            263, 249, 390, 373, 374, 380, 381, 382, 362,  # Lower
            # Eye corners
            33, 133, 362, 263
        ]
        landmarks.update(eyes)
        
        return sorted(list(landmarks))
    
    def _get_expanded_names(self):
        """Generate names for expanded landmarks."""
        # Create descriptive names for the landmarks
        landmark_names = {}
        
        # Magic 6 names
        landmark_names.update({
            1: 'noseTip', 2: 'noseBottom', 98: 'noseRightCorner',
            327: 'noseLeftCorner', 205: 'rightCheek', 425: 'leftCheek'
        })
        
        # Add generic names for other landmarks
        for landmark_idx in self.feature_landmarks:
            if landmark_idx not in landmark_names:
                if landmark_idx in [161, 185, 40, 39, 37, 0, 267, 269, 270, 409, 29]:
                    landmark_names[landmark_idx] = f'upperLip_{landmark_idx}'
                elif landmark_idx in [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]:
                    landmark_names[landmark_idx] = f'lowerLip_{landmark_idx}'
                elif landmark_idx in [78, 13, 308, 14, 317, 402]:
                    landmark_names[landmark_idx] = f'innerLip_{landmark_idx}'
                elif landmark_idx in [246, 161, 160, 159, 158, 157, 173, 33, 7, 163, 144, 145, 153, 154, 155, 133]:
                    landmark_names[landmark_idx] = f'rightEye_{landmark_idx}'
                elif landmark_idx in [466, 388, 387, 386, 385, 384, 398, 263, 249, 390, 373, 374, 380, 381, 382, 362]:
                    landmark_names[landmark_idx] = f'leftEye_{landmark_idx}'
                else:
                    landmark_names[landmark_idx] = f'landmark_{landmark_idx}'
        
        return [landmark_names[idx] for idx in self.feature_landmarks]
        
    def extract_rotation_points(self, df, frame_idx, scale_z=True):
        """
        Extract only the Magic 6 landmarks for Kabsch rotation computation.
        
        Args:
            df: DataFrame with feature columns
            frame_idx: Row index to extract
            scale_z: Whether to scale z coordinates by Face Depth
            
        Returns:
            numpy array of shape (6, 3) - Magic 6 landmarks only
        """
        points = []
        
        # Get Face Depth for this frame if scaling
        face_depth = df.iloc[frame_idx]['Face Depth (cm)'] if scale_z else 1.0
        
        # Extract only Magic 6 landmarks for rotation
        for landmark_idx in self.rotation_landmarks:
            x = df.iloc[frame_idx][f'feat_{landmark_idx}_x']
            y = df.iloc[frame_idx][f'feat_{landmark_idx}_y']
            z = df.iloc[frame_idx][f'feat_{landmark_idx}_z'] * face_depth
            points.append([x, y, z])
        
        return np.array(points)
    
    def extract_feature_points(self, df, frame_idx, scale_z=True):
        """
        Extract expanded landmark set for feature generation.
        
        Args:
            df: DataFrame with feature columns
            frame_idx: Row index to extract
            scale_z: Whether to scale z coordinates by Face Depth
            
        Returns:
            numpy array of shape (N, 3) - expanded landmark set
        """
        points = []
        
        # Get Face Depth for this frame if scaling
        face_depth = df.iloc[frame_idx]['Face Depth (cm)'] if scale_z else 1.0
        
        # Extract expanded landmark set
        for landmark_idx in self.feature_landmarks:
            x = df.iloc[frame_idx][f'feat_{landmark_idx}_x']
            y = df.iloc[frame_idx][f'feat_{landmark_idx}_y']
            z = df.iloc[frame_idx][f'feat_{landmark_idx}_z'] * face_depth
            points.append([x, y, z])
        
        return np.array(points)
    
    def compute_rolling_baseline_rotation(self, df, frame_idx, window_size=5, scale_z=True):
        """
        Compute rolling baseline using only Magic 6 landmarks for rotation.
        
        Args:
            df: DataFrame with feature columns
            frame_idx: Current frame index
            window_size: Size of rolling window (default: 5)
            scale_z: Whether to scale z coordinates
            
        Returns:
            numpy array of shape (6, 3) representing rotation baseline
        """
        # Determine window bounds
        start_idx = max(0, frame_idx - window_size + 1)
        end_idx = frame_idx + 1
        
        # Extract Magic 6 points for each frame in window
        window_points = []
        for idx in range(start_idx, end_idx):
            points = self.extract_rotation_points(df, idx, scale_z)
            window_points.append(points)
        
        # Compute average baseline
        baseline_points = np.mean(window_points, axis=0)
        return baseline_points
    
    def compute_rolling_baseline_features(self, df, frame_idx, window_size=5, scale_z=True):
        """
        Compute rolling baseline using expanded landmark set for features.
        
        Args:
            df: DataFrame with feature columns
            frame_idx: Current frame index
            window_size: Size of rolling window (default: 5)
            scale_z: Whether to scale z coordinates
            
        Returns:
            numpy array of shape (N, 3) representing feature baseline
        """
        # Determine window bounds
        start_idx = max(0, frame_idx - window_size + 1)
        end_idx = frame_idx + 1
        
        # Extract expanded points for each frame in window
        window_points = []
        for idx in range(start_idx, end_idx):
            points = self.extract_feature_points(df, idx, scale_z)
            window_points.append(points)
        
        # Compute average baseline
        baseline_points = np.mean(window_points, axis=0)
        return baseline_points
    
    def process_session_data(self, df, session_name, participant_id, window_size=5):
        """
        Process session data to create expanded feature set.
        
        Uses Magic 6 for rotation computation, expanded set for features:
        - Raw coordinates (N landmarks × 3 axes)
        - Baseline coordinates (N landmarks × 3 axes)  
        - Transformed coordinates (N landmarks × 3 axes)
        Total: N landmarks × 3 coordinate sets × 3 axes
        
        Args:
            df: DataFrame with raw session data
            session_name: Name of the session
            participant_id: Participant identifier
            window_size: Rolling window size
            
        Returns:
            DataFrame with processed features
        """
        print(f"  Processing expanded landmark set with stable rotation...")
        
        # Initialize results storage
        processed_data = []
        
        # Process each frame
        for frame_idx in tqdm(range(len(df)), desc=f"    Processing {len(df)} frames", leave=False):
            
            # Extract current frame points
            rotation_points = self.extract_rotation_points(df, frame_idx, scale_z=True)  # Magic 6 for rotation
            feature_points = self.extract_feature_points(df, frame_idx, scale_z=True)   # Expanded set for features
            
            # Compute rolling baselines
            rotation_baseline = self.compute_rolling_baseline_rotation(df, frame_idx, window_size, scale_z=True)
            feature_baseline = self.compute_rolling_baseline_features(df, frame_idx, window_size, scale_z=True)
            
            # Apply Kabsch transformation using ONLY Magic 6 landmarks
            R, t, rmsd = KabschAlgorithm.kabsch_algorithm(rotation_baseline, rotation_points)
            
            # Apply the same transformation to ALL feature landmarks
            transformed_features = KabschAlgorithm.apply_transformation(feature_points, R, t)
            
            # Create feature row
            row_data = {
                'participant_id': participant_id,
                'session_name': session_name,
                'frame_idx': frame_idx,
                'rmsd': rmsd
            }
            
            # Add raw, baseline, and transformed coordinates for each feature landmark
            for i, (landmark_idx, landmark_name) in enumerate(zip(self.feature_landmarks, self.feature_names)):
                # Raw coordinates
                row_data[f'{landmark_name}_raw_x'] = feature_points[i, 0]
                row_data[f'{landmark_name}_raw_y'] = feature_points[i, 1]
                row_data[f'{landmark_name}_raw_z'] = feature_points[i, 2]
                
                # Baseline coordinates
                row_data[f'{landmark_name}_baseline_x'] = feature_baseline[i, 0]
                row_data[f'{landmark_name}_baseline_y'] = feature_baseline[i, 1]
                row_data[f'{landmark_name}_baseline_z'] = feature_baseline[i, 2]
                
                # Transformed coordinates (aligned using Magic 6 rotation)
                row_data[f'{landmark_name}_transformed_x'] = transformed_features[i, 0]
                row_data[f'{landmark_name}_transformed_y'] = transformed_features[i, 1]
                row_data[f'{landmark_name}_transformed_z'] = transformed_features[i, 2]
            
            # Add any existing target/label columns
            for col in ['session_number', 'session', 'expression', 'target', 'label']:
                if col in df.columns:
                    row_data[col] = df.iloc[frame_idx][col]
            
            processed_data.append(row_data)
        
        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_data)
        
        # Create time-based targets if no target column exists
        if not any(col in processed_df.columns for col in ['session_number', 'session', 'expression', 'target', 'label']):
            n_frames = len(processed_df)
            if n_frames > 20:  # Only if enough frames
                # Create 4 time-based segments as targets
                time_segments = pd.cut(range(n_frames), bins=4, labels=[0, 1, 2, 3])
                processed_df['time_segment'] = time_segments
                print(f"  🎯 Created time-based targets: 4 segments from {n_frames} frames")
        
        return processed_df

def analyze_participant_sessions(participant_dir):
    """
    Analyze all raw session files for a participant and combine into one dataset.
    
    Args:
        participant_dir: Path to participant directory
        
    Returns:
        Dictionary with results
    """
    participant_dir = Path(participant_dir)
    participant_id = participant_dir.name
    
    print(f"\n{'='*60}")
    print(f"PROCESSING PARTICIPANT: {participant_id.upper()}")
    print(f"{'='*60}")
    
    # Find RAW session files (exclude -rb derived files)
    session_files = []
    participant_prefix = f"{participant_id}-"
    
    # Look for RAW files with participant prefix
    for pattern in [f'{participant_prefix}session*.csv', f'{participant_prefix}baseline.csv']:
        candidate_files = list(participant_dir.glob(pattern))
        # Filter out files with "-rb" in their names (derived features)
        raw_files = [f for f in candidate_files if '-rb' not in f.name]
        session_files.extend(raw_files)
    
    if not session_files:
        print(f"❌ No raw session files found in {participant_dir}")
        return {}
    
    print(f"📁 Found {len(session_files)} raw session files")
    
    processor = FocusedKabschProcessor()
    all_processed_data = []  # Will hold all frames from all sessions
    session_stats = {}
    
    # Process each session and collect all frames
    for session_file in sorted(session_files):
        session_name = session_file.stem
        print(f"\n📊 Processing: {session_name}")
        
        try:
            # Load raw session data
            df = pd.read_csv(session_file)
            print(f"  📈 Loaded {len(df)} frames")
            
            # Process session data (Magic 6 landmarks with 3 coordinate sets each)
            processed_df = processor.process_session_data(df, session_name, participant_id)
            
            # Add session label for SVM prediction
            processed_df['target_session'] = session_name
            
            # Append to combined dataset
            all_processed_data.append(processed_df)
            
            session_stats[session_name] = {
                'n_frames': len(processed_df),
                'status': 'success'
            }
            
            print(f"  ✅ Processed {len(processed_df)} frames")
            
        except Exception as e:
            print(f"  ❌ Error processing {session_name}: {str(e)}")
            session_stats[session_name] = {
                'n_frames': 0,
                'status': 'error',
                'error': str(e)
            }
    
    if not all_processed_data:
        print("❌ No sessions were successfully processed")
        return {'error': 'No data processed'}
    
    # Combine all sessions into one dataset
    print(f"\n🔗 Combining all sessions into one dataset...")
    combined_df = pd.concat(all_processed_data, ignore_index=True)
    
    # Save consolidated training data
    datasets_path = Path("../../training/datasets")
    datasets_path.mkdir(exist_ok=True)
    
    output_filename = f"{participant_id}_all_sessions_expanded_kabsch.csv"
    output_path = datasets_path / output_filename
    
    combined_df.to_csv(output_path, index=False)
    
    # Report dataset statistics
    feature_cols = [col for col in combined_df.columns if any(x in col for x in ['_raw_', '_baseline_', '_transformed_'])]
    n_features = len(feature_cols)
    n_samples = len(combined_df)
    n_sessions = combined_df['target_session'].nunique()
    
    print(f"💾 Saved consolidated dataset to: {output_path}")
    print(f"📊 Dataset: {n_samples} total frames × {n_features} features")
    print(f"🎯 Target: {n_sessions} different sessions")
    
    # Test SVM performance on combined dataset
    print(f"\n🤖 Testing SVM on combined dataset...")
    svm_results = test_svm_on_combined_data(combined_df)
    
    if 'error' not in svm_results:
        print(f"  🎯 SVM Accuracy: {svm_results['accuracy_mean']:.3f} ± {svm_results['accuracy_std']:.3f}")
        print(f"  📊 Predicting {svm_results['n_classes']} sessions from {svm_results['n_features']} features")
    else:
        print(f"  ⚠️ SVM test failed: {svm_results['error']}")
    
    return {
        'participant_id': participant_id,
        'output_file': output_path,
        'session_stats': session_stats,
        'combined_stats': {
            'total_frames': n_samples,
            'n_features': n_features,
            'n_sessions': n_sessions
        },
        'svm_results': svm_results
    }

def test_svm_on_combined_data(combined_df, cv_folds=5):
    """
    Test SVM performance on combined dataset to predict session names.
    
    Args:
        combined_df: DataFrame with all frames from all sessions
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with performance metrics
    """
    # Use session name as target
    if 'target_session' not in combined_df.columns:
        return {'error': 'No target_session column found'}
    
    # Get feature columns (raw, baseline, transformed coordinates)
    feature_cols = [col for col in combined_df.columns if any(x in col for x in ['_raw_', '_baseline_', '_transformed_'])]
    
    if not feature_cols:
        return {'error': 'No feature columns found'}
    
    # Prepare data
    X = combined_df[feature_cols].values
    y = combined_df['target_session'].values
    
    # Check if we have multiple sessions
    unique_sessions = np.unique(y)
    if len(unique_sessions) < 2:
        return {'error': f'Need at least 2 sessions, found {len(unique_sessions)}'}
    
    # Test SVM with proper scaling
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Create SVM pipeline with scaling
    from sklearn.pipeline import Pipeline
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
    ])
    
    # Perform cross-validation
    scores = cross_val_score(svm_pipeline, X, y, cv=kfold, scoring='accuracy')
    
    results = {
        'target_column': 'target_session',
        'n_features': len(feature_cols),
        'n_samples': len(X),
        'n_classes': len(unique_sessions),
        'session_names': unique_sessions.tolist(),
        'accuracy_mean': scores.mean(),
        'accuracy_std': scores.std(),
        'cv_scores': scores.tolist()
    }
    
    return results

def print_summary(results, participant_id):
    """
    Print summary of processing results.
    
    Args:
        results: Dictionary with processing results
        participant_id: Participant identifier
    """
    print(f"\n{'='*80}")
    print(f"SUMMARY: {participant_id.upper()}")
    print(f"{'='*80}")
    
    if 'error' in results:
        print(f"❌ Processing failed: {results['error']}")
        return
    
    # Session processing summary
    print(f"\n📊 SESSION PROCESSING:")
    print(f"{'Session':<20} {'Frames':<8} {'Status'}")
    print(f"{'-'*35}")
    
    total_frames = 0
    successful_sessions = 0
    
    for session_name, stats in results['session_stats'].items():
        if stats['status'] == 'success':
            print(f"{session_name:<20} {stats['n_frames']:<8} {'✅ Success'}")
            total_frames += stats['n_frames']
            successful_sessions += 1
        else:
            print(f"{session_name:<20} {'0':<8} {'❌ Failed'}")
    
    # Combined dataset summary
    combined_stats = results['combined_stats']
    print(f"\n🔗 COMBINED DATASET:")
    print(f"  • Total Frames: {combined_stats['total_frames']}")
    print(f"  • Features per Frame: {combined_stats['n_features']}")
    print(f"  • Target Sessions: {combined_stats['n_sessions']}")
    print(f"  • Successful Sessions: {successful_sessions}")
    
    # SVM performance
    svm_results = results['svm_results']
    if 'error' not in svm_results:
        print(f"\n🤖 SVM PERFORMANCE (Session Prediction):")
        print(f"  • Mean Accuracy: {svm_results['accuracy_mean']:.3f} ± {svm_results['accuracy_std']:.3f}")
        print(f"  • Predicting: {svm_results['n_classes']} different sessions")
        print(f"  • Feature Efficiency: {svm_results['accuracy_mean']/svm_results['n_features']*100:.2f}% per feature")
        print(f"  • Sessions: {', '.join(svm_results['session_names'])}")
    else:
        print(f"\n⚠️ SVM TESTING FAILED: {svm_results['error']}")
    
    print(f"\n💾 OUTPUT:")
    print(f"  • File: {results['output_file']}")
    print(f"  • Ready for cross-participant training")
    print(f"  • Target column: 'target_session'")
    print(f"  • Features: {results['combined_stats']['n_features']} (expanded landmarks × 3 coordinate sets)")

def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python optimized_kabsch_stable_rotation.py <participant_directory>")
        print("Example: python optimized_kabsch_stable_rotation.py ../../read/e1")
        sys.exit(1)
    
    participant_dir = sys.argv[1]
    
    if not os.path.exists(participant_dir):
        print(f"❌ Directory not found: {participant_dir}")
        sys.exit(1)
    
    print("🔬 FOCUSED KABSCH ALIGNMENT FOR SVM TRAINING")
    print("=" * 60)
    print("This script:")
    print("• Uses Magic 6 landmarks for stable Kabsch rotation: [1, 2, 98, 327, 205, 425]")
    print("• Extracts features from expanded set: nose + cheeks + eyes + lips")
    print("• Creates 3 coordinate sets per landmark: raw, baseline, transformed")
    print("• Applies Z-scaling (z *= face_depth) before processing")
    print("• Saves compact training data to training/datasets/ folder")
    print("• Tests SVM performance with StandardScaler + 5-fold cross-validation")
    
    # Process participant
    participant_path = Path(participant_dir)
    all_results = analyze_participant_sessions(participant_path)
    
    # Print summary
    if all_results:
        print_summary(all_results, participant_path.name)
    else:
        print("❌ No results to display")

if __name__ == "__main__":
    main() 