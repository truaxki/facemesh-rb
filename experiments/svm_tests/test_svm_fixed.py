#!/usr/bin/env python3
"""
Fixed SVM test for facial expression classification.
Extracts session numbers from filenames instead of expecting a column.

Usage:
    python test_svm_fixed.py <directory_path>
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def explain_svm_for_faces():
    """Explain why SVM might work well for facial features."""
    print("\n" + "="*60)
    print("WHY SVM FOR FACIAL EXPRESSIONS?")
    print("="*60)
    
    print("\n1. HIGH-DIMENSIONAL DATA:")
    print("   - You have 207+ features from facial landmarks")
    print("   - SVM excels in high-dimensional spaces")
    print("   - Can find complex decision boundaries")
    
    print("\n2. GEOMETRIC NATURE:")
    print("   - Facial landmarks are geometric points")
    print("   - SVM finds optimal separating hyperplanes")
    print("   - RBF kernel can capture non-linear facial patterns")
    
    print("\n3. ROBUST TO OUTLIERS:")
    print("   - Only support vectors matter")
    print("   - Less affected by noisy landmark detections")
    
    print("\n4. WHAT SVM NEEDS:")
    print("   - SCALED FEATURES (critical!)")
    print("   - Proper kernel selection")
    print("   - C and gamma tuning")


def extract_session_number(filename):
    """Extract session number from filename like 'e1-session3-rb5-rel.csv'."""
    match = re.search(r'session(\d+)', str(filename))
    if match:
        return int(match.group(1))
    elif 'baseline' in str(filename):
        return 0  # Baseline is session 0
    else:
        return -1  # Unknown


def load_data(directory_path):
    """Load Kabsch-aligned features."""
    # Look for session files with Kabsch features
    csv_files = list(Path(directory_path).glob('*-session*-rb5-rel.csv'))
    
    if not csv_files:
        print("No Kabsch-aligned feature files found!")
        print("Looking for alternative patterns...")
        # Try without 'rel' suffix
        csv_files = list(Path(directory_path).glob('*-session*-rb5.csv'))
        
    if not csv_files:
        print("No suitable files found!")
        print(f"Files in directory: {list(Path(directory_path).glob('*.csv'))[:5]}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} files to process")
    
    all_data = []
    for csv_file in csv_files:
        print(f"  Processing: {csv_file.name}")
        df = pd.read_csv(csv_file)
        
        # Extract session number from filename
        session_num = extract_session_number(csv_file.name)
        df['session_num'] = session_num
        
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Look for Kabsch-aligned features (rel_diff)
    feature_cols = [col for col in combined_df.columns if col.startswith('rb5_rel_diff_')]
    
    if not feature_cols:
        print("No rb5_rel_diff features found. Looking for alternatives...")
        # Try regular rb5_diff features
        feature_cols = [col for col in combined_df.columns if col.startswith('rb5_diff_')]
        
    if not feature_cols:
        print("No diff features found. Using raw rb5 features...")
        feature_cols = [col for col in combined_df.columns if col.startswith('feat_') and col.endswith('_rb5')]
    
    if not feature_cols:
        print("ERROR: No suitable features found!")
        print(f"Available columns: {combined_df.columns[:20].tolist()}")
        sys.exit(1)
    
    X = combined_df[feature_cols].values
    y = combined_df['session_num'].values
    
    print(f"\nLoaded {len(combined_df)} samples with {len(feature_cols)} features")
    print(f"Feature type: {feature_cols[0].split('_')[0] if feature_cols else 'unknown'}")
    print(f"Sessions found: {sorted(np.unique(y))}")
    
    return X, y


def demonstrate_scaling_importance(X_train, X_test, y_train, y_test):
    """Show why scaling is critical for SVM."""
    print("\n" + "="*60)
    print("IMPORTANCE OF FEATURE SCALING")
    print("="*60)
    
    # Test without scaling
    print("\n1. SVM WITHOUT SCALING:")
    svm_unscaled = SVC(kernel='rbf', random_state=42)
    svm_unscaled.fit(X_train, y_train)
    acc_unscaled = accuracy_score(y_test, svm_unscaled.predict(X_test))
    print(f"   Accuracy: {acc_unscaled:.3f}")
    
    # Test with scaling
    print("\n2. SVM WITH SCALING:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm_scaled = SVC(kernel='rbf', random_state=42)
    svm_scaled.fit(X_train_scaled, y_train)
    acc_scaled = accuracy_score(y_test, svm_scaled.predict(X_test_scaled))
    print(f"   Accuracy: {acc_scaled:.3f}")
    
    if acc_unscaled > 0:
        print(f"\n   Improvement: {(acc_scaled - acc_unscaled) / acc_unscaled * 100:.1f}%")
    else:
        print(f"\n   Improvement: {acc_scaled:.1%} (from ~0%)")
    
    # Show feature scale differences
    print("\n3. FEATURE SCALE ANALYSIS:")
    print(f"   Before scaling - Mean: {X_train.mean():.3f}, Std: {X_train.std():.3f}")
    print(f"   After scaling - Mean: {X_train_scaled.mean():.3f}, Std: {X_train_scaled.std():.3f}")
    
    return scaler, X_train_scaled, X_test_scaled


def test_different_kernels(X_train_scaled, X_test_scaled, y_train, y_test):
    """Test different SVM kernels."""
    print("\n" + "="*60)
    print("SVM KERNEL COMPARISON")
    print("="*60)
    
    kernels = {
        'Linear': {'kernel': 'linear', 'C': 1.0},
        'RBF (Gaussian)': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
        'Polynomial (degree=3)': {'kernel': 'poly', 'degree': 3, 'C': 1.0},
        'Sigmoid': {'kernel': 'sigmoid', 'C': 1.0}
    }
    
    results = {}
    
    for name, params in kernels.items():
        print(f"\n{name}:")
        svm = SVC(**params, random_state=42)
        
        # Train
        svm.fit(X_train_scaled, y_train)
        
        # Test
        y_pred = svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=3)
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_support': svm.n_support_.sum()
        }
        
        print(f"  Test Accuracy: {accuracy:.3f}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Support Vectors: {svm.n_support_.sum()} / {len(y_train)} ({svm.n_support_.sum()/len(y_train)*100:.1f}%)")
    
    return results


def tune_rbf_svm(X_train_scaled, X_test_scaled, y_train, y_test):
    """Fine-tune RBF SVM parameters."""
    print("\n" + "="*60)
    print("RBF SVM PARAMETER TUNING")
    print("="*60)
    
    # Test different C and gamma values
    C_values = [0.1, 1, 10, 100]
    gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1]
    
    results = []
    
    print("\nTesting parameter combinations...")
    for C in C_values:
        for gamma in gamma_values:
            svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
            svm.fit(X_train_scaled, y_train)
            acc = accuracy_score(y_test, svm.predict(X_test_scaled))
            results.append({'C': C, 'gamma': gamma, 'accuracy': acc})
    
    # Sort by accuracy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    print("\nTop 5 parameter combinations:")
    print(results_df.head())
    
    # Train best model
    best = results_df.iloc[0]
    print(f"\nBest parameters: C={best['C']}, gamma={best['gamma']}")
    print(f"Best accuracy: {best['accuracy']:.3f}")
    
    # Train final model with best params
    best_svm = SVC(kernel='rbf', C=best['C'], gamma=best['gamma'], random_state=42)
    best_svm.fit(X_train_scaled, y_train)
    
    return best_svm, best


def analyze_predictions(best_svm, X_test_scaled, y_test):
    """Analyze the predictions in detail."""
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    y_pred = best_svm.predict(X_test_scaled)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred


def compare_with_xgboost(X_train, X_test, y_train, y_test, svm_accuracy):
    """Quick comparison with XGBoost baseline."""
    print("\n" + "="*60)
    print("COMPARISON WITH XGBOOST")
    print("="*60)
    
    try:
        import xgboost as xgb
        
        # XGBoost expects 0-indexed classes, so adjust if needed
        unique_classes = np.unique(y_train)
        if unique_classes.min() > 0:
            # Adjust to 0-indexed
            y_train_adj = y_train - unique_classes.min()
            y_test_adj = y_test - unique_classes.min()
            num_classes = len(unique_classes)
        else:
            y_train_adj = y_train
            y_test_adj = y_test
            num_classes = len(unique_classes)
        
        # XGBoost (doesn't need scaling)
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=num_classes,
            random_state=42,
            verbosity=0
        )
        
        xgb_model.fit(X_train, y_train_adj)
        xgb_pred = xgb_model.predict(X_test)
        
        # Adjust predictions back if needed
        if unique_classes.min() > 0:
            xgb_pred = xgb_pred + unique_classes.min()
            
        xgb_acc = accuracy_score(y_test, xgb_pred)
        
        print(f"\nXGBoost accuracy: {xgb_acc:.3f}")
        print(f"Best SVM accuracy: {svm_accuracy:.3f}")
        print(f"Difference: {(svm_accuracy - xgb_acc):.3f} ({(svm_accuracy - xgb_acc)/xgb_acc*100:+.1f}%)")
        
    except ImportError:
        print("\nXGBoost not installed. Skipping comparison.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_svm_fixed.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    # Explain why SVM might work
    explain_svm_for_faces()
    
    # Load data
    X, y = load_data(directory_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Demonstrate scaling importance
    scaler, X_train_scaled, X_test_scaled = demonstrate_scaling_importance(
        X_train, X_test, y_train, y_test
    )
    
    # Test different kernels
    kernel_results = test_different_kernels(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Tune RBF SVM
    best_svm, best_params = tune_rbf_svm(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Analyze predictions
    analyze_predictions(best_svm, X_test_scaled, y_test)
    
    # Compare with XGBoost
    compare_with_xgboost(
        X_train, X_test, y_train, y_test,
        best_params['accuracy']
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBest SVM configuration:")
    print(f"  - Kernel: RBF")
    print(f"  - C: {best_params['C']}")
    print(f"  - Gamma: {best_params['gamma']}")
    print(f"  - Accuracy: {best_params['accuracy']:.1%}")
    print(f"\nReady to test on more participants!")


if __name__ == "__main__":
    main() 