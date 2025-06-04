"""
Template script for loading facemesh data with rolling baseline features
for machine learning training.

This script provides a starting point for creating training datasets
from the enhanced CSV files.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FacemeshDataLoader:
    """
    Data loader for facemesh data with rolling baseline features.
    
    This class handles:
    - Loading enhanced CSV files (rb5 or rb10)
    - Feature selection
    - Train/validation/test splitting
    - Data normalization
    - Label preparation
    """
    
    def __init__(self, data_root: str = "../../read", window_size: int = 5):
        """
        Initialize the data loader.
        
        Args:
            data_root: Root directory containing subject folders
            window_size: Rolling baseline window size (5 or 10)
        """
        self.window_size = window_size
        self.suffix = f"-rb{window_size}"
        
        # Try to find the correct data directory
        candidate_paths = [data_root, "read", "../read", "../../read", "../../../read"]
        self.data_root = None
        
        for path in candidate_paths:
            test_path = Path(path)
            if test_path.exists() and test_path.is_dir():
                # Check if it contains subject directories (e.g., e1, e2, etc.)
                if any(d.is_dir() and d.name.startswith('e') and d.name[1:].isdigit() 
                      for d in test_path.iterdir()):
                    self.data_root = test_path
                    print(f"Found data directory at: {self.data_root.absolute()}")
                    break
        
        if self.data_root is None:
            self.data_root = Path(data_root)  # Fall back to original path
            print(f"Warning: Could not find data directory. Using: {self.data_root.absolute()}")
            print(f"Tried paths: {[str(Path(p).absolute()) for p in candidate_paths]}")
        
    def load_subject_data(self, subject_id: str, session: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for a specific subject and optional session.
        
        Args:
            subject_id: Subject identifier (e.g., 'e1', 'e2', etc.)
            session: Optional session name (e.g., 'baseline', 'session1')
                    If None, loads all sessions for the subject
        
        Returns:
            DataFrame with loaded data
        """
        subject_path = self.data_root / subject_id
        
        if session:
            file_pattern = f"{subject_id}-{session}{self.suffix}.csv"
            file_path = subject_path / file_pattern
            if file_path.exists():
                return pd.read_csv(file_path)
            else:
                print(f"Warning: File not found - {file_path}")
                return pd.DataFrame()
        else:
            # Load all sessions for the subject
            all_data = []
            for csv_file in subject_path.glob(f"*{self.suffix}.csv"):
                df = pd.read_csv(csv_file)
                # Add session identifier from filename
                session_name = csv_file.stem.replace(f"{subject_id}-", "").replace(self.suffix, "")
                df['Session'] = session_name
                all_data.append(df)
            
            if all_data:
                return pd.concat(all_data, ignore_index=True)
            else:
                print(f"Warning: No data found for subject {subject_id}")
                return pd.DataFrame()
    
    def select_features(self, df: pd.DataFrame, 
                       feature_types: List[str] = ['rb', 'diff']) -> pd.DataFrame:
        """
        Select specific types of features from the dataframe.
        
        Args:
            df: Input dataframe
            feature_types: Types of features to select
                          Options: 'base', 'rb', 'diff', 'metadata'
        
        Returns:
            DataFrame with selected features
        """
        selected_cols = []
        
        # Always include metadata
        metadata_cols = ['Subject Name', 'Test Name', 'Time (s)', 'Session']
        selected_cols.extend([col for col in metadata_cols if col in df.columns])
        
        if 'base' in feature_types:
            # Original coordinate features
            base_cols = [col for col in df.columns 
                        if col.startswith('feat_') and 
                        not any(suffix in col for suffix in ['_rb', '_diff'])]
            selected_cols.extend(base_cols)
        
        if 'rb' in feature_types:
            # Rolling baseline averages
            rb_cols = [col for col in df.columns 
                      if f'_rb{self.window_size}' in col and not col.endswith('_diff')]
            selected_cols.extend(rb_cols)
        
        if 'diff' in feature_types:
            # Baseline deviation features
            diff_cols = [col for col in df.columns 
                        if col.endswith(f'_rb{self.window_size}_diff')]
            selected_cols.extend(diff_cols)
        
        return df[selected_cols]
    
    def prepare_sequences(self, df: pd.DataFrame, 
                         sequence_length: int = 30,
                         stride: int = 10) -> np.ndarray:
        """
        Convert time series data into sequences for temporal models.
        
        Args:
            df: Input dataframe (should be from a single session)
            sequence_length: Length of each sequence
            stride: Step size between sequences
        
        Returns:
            Array of shape (n_sequences, sequence_length, n_features)
        """
        # Remove metadata columns for sequence creation
        feature_cols = [col for col in df.columns 
                       if col.startswith('feat_') or '_rb' in col]
        
        data = df[feature_cols].values
        sequences = []
        
        for i in range(0, len(data) - sequence_length + 1, stride):
            sequences.append(data[i:i + sequence_length])
        
        return np.array(sequences)
    
    def load_dataset(self, subjects: List[str], 
                    sessions: Optional[List[str]] = None,
                    test_size: float = 0.2,
                    val_size: float = 0.2,
                    random_state: int = 42) -> Dict:
        """
        Load complete dataset with train/validation/test split.
        
        Args:
            subjects: List of subject IDs to include
            sessions: Optional list of sessions to include
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        all_data = []
        
        for subject in subjects:
            if sessions:
                for session in sessions:
                    df = self.load_subject_data(subject, session)
                    if not df.empty:
                        all_data.append(df)
            else:
                df = self.load_subject_data(subject)
                if not df.empty:
                    all_data.append(df)
        
        if not all_data:
            raise ValueError("No data loaded for specified subjects/sessions")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create train/test split by subject to avoid data leakage
        unique_subjects = combined_df['Subject Name'].unique()
        train_val_subjects, test_subjects = train_test_split(
            unique_subjects, test_size=test_size, random_state=random_state
        )
        
        # Further split train/val
        train_subjects, val_subjects = train_test_split(
            train_val_subjects, test_size=val_size, random_state=random_state
        )
        
        # Create splits
        train_df = combined_df[combined_df['Subject Name'].isin(train_subjects)]
        val_df = combined_df[combined_df['Subject Name'].isin(val_subjects)]
        test_df = combined_df[combined_df['Subject Name'].isin(test_subjects)]
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'subjects': {
                'train': list(train_subjects),
                'val': list(val_subjects),
                'test': list(test_subjects)
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = FacemeshDataLoader(window_size=5)  # Uses default path now
    
    # Example 1: Load single subject data
    print("Loading data for subject e1...")
    df_e1 = loader.load_subject_data("e1", "baseline")
    print(f"Loaded {len(df_e1)} rows, {len(df_e1.columns)} columns")
    
    # Example 2: Select specific features
    print("\nSelecting baseline deviation features...")
    df_diff = loader.select_features(df_e1, feature_types=['diff'])
    print(f"Selected {len(df_diff.columns)} columns")
    
    # Example 3: Create sequences for temporal models
    if not df_e1.empty:
        print("\nCreating sequences...")
        sequences = loader.prepare_sequences(df_e1, sequence_length=30, stride=10)
        print(f"Created {sequences.shape[0]} sequences of shape {sequences.shape[1:]}")
    
    # Example 4: Load complete dataset
    print("\nLoading dataset for multiple subjects...")
    subjects = [f"e{i}" for i in range(1, 6)]  # e1 to e5
    try:
        dataset = loader.load_dataset(subjects, sessions=['baseline', 'session1'])
        print(f"Train: {len(dataset['train'])} rows")
        print(f"Val: {len(dataset['val'])} rows")
        print(f"Test: {len(dataset['test'])} rows")
        print(f"Train subjects: {dataset['subjects']['train']}")
    except Exception as e:
        print(f"Error loading dataset: {e}") 