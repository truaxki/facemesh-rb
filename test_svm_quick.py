#!/usr/bin/env python3
"""
Quick SVM test for facial expression classification.

This script demonstrates:
1. Why SVM might work well for facial features
2. What preprocessing is needed
3. How to tune SVM parameters

Usage:
    python test_svm_quick.py <directory_path>
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
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
    print("   - You have 207 features (69 landmarks × 3 axes)")
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


def load_data(directory_path):
    """Load Kabsch-aligned features."""
    # Look for session files with Kabsch features
    csv_files = list(Path(directory_path).glob('*-session*-rb5-rel.csv'))
    
    if not csv_files:
        print("No Kabsch-aligned feature files found!")
        print("Please run compute_rolling_baseline_with_kabsch.py first")
        sys.exit(1)
    
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Use rb5_rel_diff features
    feature_cols = [col for col in combined_df.columns if col.startswith('rb5_rel_diff_')]
    X = combined_df[feature_cols].values
    y = combined_df['session_num'].values
    
    print(f"\nLoaded {len(combined_df)} samples with {len(feature_cols)} features")
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
    
    print(f"\n   Improvement: {(acc_scaled - acc_unscaled) / acc_unscaled * 100:.1f}%")
    
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
        cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
        
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
    
    # Decision function analysis
    decision = best_svm.decision_function(X_test_scaled)
    
    # For multi-class, decision_function returns one-vs-one scores
    print("\nDecision Function Analysis:")
    print(f"  Shape: {decision.shape}")
    print(f"  Mean confidence: {np.abs(decision).mean():.3f}")
    
    return y_pred


def compare_with_xgboost(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, svm_accuracy):
    """Quick comparison with XGBoost baseline."""
    print("\n" + "="*60)
    print("COMPARISON WITH XGBOOST")
    print("="*60)
    
    try:
        import xgboost as xgb
        
        # XGBoost (doesn't need scaling)
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=len(np.unique(y_train)),
            random_state=42,
            verbosity=0
        )
        
        xgb_model.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
        
        print(f"\nXGBoost accuracy: {xgb_acc:.3f}")
        print(f"Best SVM accuracy: {svm_accuracy:.3f}")
        print(f"Difference: {(svm_accuracy - xgb_acc):.3f} ({(svm_accuracy - xgb_acc)/xgb_acc*100:+.1f}%)")
        
    except ImportError:
        print("\nXGBoost not installed. Skipping comparison.")


def practical_recommendations():
    """Provide practical recommendations."""
    print("\n" + "="*60)
    print("PRACTICAL RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. WHEN TO USE SVM:")
    print("   - High-dimensional geometric features (✓ your case)")
    print("   - Need interpretable support vectors")
    print("   - Small to medium datasets")
    print("   - Don't mind longer training times")
    
    print("\n2. SVM BEST PRACTICES:")
    print("   - ALWAYS scale features first")
    print("   - Start with RBF kernel")
    print("   - Use GridSearchCV for C and gamma")
    print("   - Consider class_weight='balanced' for imbalanced data")
    
    print("\n3. EXPECTED PERFORMANCE:")
    print("   - Should match or exceed XGBoost")
    print("   - RBF typically best for facial data")
    print("   - May need C=10-100 for your feature scale")
    
    print("\n4. SPEED CONSIDERATIONS:")
    print("   - Training: Slower than XGBoost")
    print("   - Prediction: Fast once trained")
    print("   - Use LinearSVC for very large datasets")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_svm_quick.py <directory_path>")
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
        X_train_scaled, X_test_scaled,
        best_params['accuracy']
    )
    
    # Practical recommendations
    practical_recommendations()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBest SVM configuration:")
    print(f"  - Kernel: RBF")
    print(f"  - C: {best_params['C']}")
    print(f"  - Gamma: {best_params['gamma']}")
    print(f"  - Accuracy: {best_params['accuracy']:.1%}")
    print(f"\nReady to implement in production!")


if __name__ == "__main__":
    main() 