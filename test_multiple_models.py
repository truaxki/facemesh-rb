#!/usr/bin/env python3
"""
Test multiple machine learning models on Kabsch-aligned features.

This script compares different models including:
- SVM (with different kernels)
- Random Forest
- Neural Networks
- LightGBM
- CatBoost
- K-Nearest Neighbors
- Ensemble methods

Usage:
    python test_multiple_models.py <directory_path> [--features <type>]
    
Example:
    python test_multiple_models.py read/e1 --features rb5_rel_diff
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Try to import optional libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed. Skipping LightGBM tests.")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed. Skipping CatBoost tests.")


def load_kabsch_features(directory_path, feature_type='rb5_rel_diff'):
    """Load Kabsch-aligned features from CSV files."""
    csv_pattern = f'p*lips_eyebrows-rb5-rel.csv'
    csv_files = list(Path(directory_path).glob(csv_pattern))
    
    if not csv_files:
        csv_pattern = f'p*-rb5-rel.csv'
        csv_files = list(Path(directory_path).glob(csv_pattern))
    
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Select features based on type
    if feature_type == 'rb5_rel_diff':
        feature_cols = [col for col in combined_df.columns if col.startswith('rb5_rel_diff_')]
    elif feature_type == 'rb5_rel_mag':
        feature_cols = [col for col in combined_df.columns if col.startswith('rb5_rel_mag_')]
    else:
        # Use all rb5 features
        feature_cols = [col for col in combined_df.columns if col.startswith('rb5_')]
    
    X = combined_df[feature_cols].values
    y = combined_df['session_num'].values
    
    print(f"\nLoaded {len(combined_df)} samples with {len(feature_cols)} features")
    print(f"Feature type: {feature_type}")
    print(f"Classes: {np.unique(y)}")
    
    return X, y, feature_cols


def test_svm_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Test different SVM configurations."""
    print("\n" + "="*60)
    print("SUPPORT VECTOR MACHINE (SVM) MODELS")
    print("="*60)
    
    svm_models = {
        'SVM Linear': SVC(kernel='linear', C=1.0, random_state=42),
        'SVM RBF': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        'SVM Polynomial': SVC(kernel='poly', degree=3, C=1.0, random_state=42),
        'SVM RBF (tuned)': None  # Will use GridSearchCV
    }
    
    results = {}
    
    for name, model in svm_models.items():
        print(f"\n{name}:")
        
        if name == 'SVM RBF (tuned)':
            # Hyperparameter tuning for RBF SVM
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            
            model = GridSearchCV(
                SVC(kernel='rbf', random_state=42),
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time
            
            print(f"  Best params: {model.best_params_}")
            y_pred = model.predict(X_test_scaled)
            
        else:
            # Train with scaled data (important for SVM!)
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_time': train_time
        }
        
        print(f"  Test Accuracy: {accuracy:.3f}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Training time: {train_time:.2f}s")
    
    return results


def test_tree_models(X_train, X_test, y_train, y_test):
    """Test tree-based models (no scaling needed)."""
    print("\n" + "="*60)
    print("TREE-BASED MODELS")
    print("="*60)
    
    tree_models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost (baseline)': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=len(np.unique(y_train)),
            random_state=42,
            verbosity=0
        ),
        'XGBoost (tuned)': xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=len(np.unique(y_train)),
            random_state=42,
            verbosity=0
        )
    }
    
    if HAS_LIGHTGBM:
        tree_models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
    
    if HAS_CATBOOST:
        tree_models['CatBoost'] = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
    
    results = {}
    
    for name, model in tree_models.items():
        print(f"\n{name}:")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_time': train_time
        }
        
        print(f"  Test Accuracy: {accuracy:.3f}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Training time: {train_time:.2f}s")
    
    return results


def test_neural_networks(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Test neural network models."""
    print("\n" + "="*60)
    print("NEURAL NETWORK MODELS")
    print("="*60)
    
    nn_models = {
        'MLP Small': MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ),
        'MLP Medium': MLPClassifier(
            hidden_layer_sizes=(200, 100),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ),
        'MLP Large': MLPClassifier(
            hidden_layer_sizes=(300, 200, 100),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    
    results = {}
    
    for name, model in nn_models.items():
        print(f"\n{name}:")
        
        # Neural networks need scaled data
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_time': train_time,
            'n_iter': model.n_iter_
        }
        
        print(f"  Test Accuracy: {accuracy:.3f}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Iterations: {model.n_iter_}")
    
    return results


def test_other_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Test other models like KNN."""
    print("\n" + "="*60)
    print("OTHER MODELS")
    print("="*60)
    
    other_models = {
        'KNN-5': KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
        'KNN-10': KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1),
        'KNN-20': KNeighborsClassifier(n_neighbors=20, weights='distance', n_jobs=-1)
    }
    
    results = {}
    
    for name, model in other_models.items():
        print(f"\n{name}:")
        
        # KNN works better with scaled data
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_time': train_time
        }
        
        print(f"  Test Accuracy: {accuracy:.3f}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Training time: {train_time:.2f}s")
    
    return results


def test_ensemble_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Test ensemble methods combining multiple models."""
    print("\n" + "="*60)
    print("ENSEMBLE MODELS")
    print("="*60)
    
    # Create base models
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        objective='multi:softmax', num_class=len(np.unique(y_train)),
        random_state=42, verbosity=0
    )
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
    
    # Voting ensemble
    voting_hard = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_model), ('svm', svm)],
        voting='hard'
    )
    
    voting_soft = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_model), ('svm', svm)],
        voting='soft'
    )
    
    results = {}
    
    # Test hard voting
    print("\nEnsemble (Hard Voting):")
    start_time = time.time()
    # Note: SVM needs scaled data, but RF/XGB don't
    # Train RF and XGB on unscaled, SVM on scaled
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    svm.fit(X_train_scaled, y_train)
    
    # Predict with each model
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    svm_pred = svm.predict(X_test_scaled)
    
    # Hard voting (majority vote)
    from scipy import stats
    y_pred = stats.mode(np.array([rf_pred, xgb_pred, svm_pred]), axis=0)[0][0]
    
    train_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    
    results['Ensemble (Hard)'] = {
        'accuracy': accuracy,
        'train_time': train_time
    }
    
    print(f"  Test Accuracy: {accuracy:.3f}")
    print(f"  Training time: {train_time:.2f}s")
    
    # Test soft voting
    print("\nEnsemble (Soft Voting):")
    start_time = time.time()
    
    # Get probability predictions
    rf_proba = rf.predict_proba(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    svm_proba = svm.predict_proba(X_test_scaled)
    
    # Average probabilities
    avg_proba = (rf_proba + xgb_proba + svm_proba) / 3
    y_pred = np.argmax(avg_proba, axis=1)
    
    train_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    
    results['Ensemble (Soft)'] = {
        'accuracy': accuracy,
        'train_time': train_time
    }
    
    print(f"  Test Accuracy: {accuracy:.3f}")
    print(f"  Training time: {train_time:.2f}s")
    
    return results


def summarize_results(all_results):
    """Create a summary table of all results."""
    print("\n" + "="*60)
    print("SUMMARY OF ALL MODELS")
    print("="*60)
    
    # Flatten results
    summary = []
    for category, models in all_results.items():
        for model_name, metrics in models.items():
            summary.append({
                'Model': model_name,
                'Category': category,
                'Test Accuracy': metrics['accuracy'],
                'CV Mean': metrics.get('cv_mean', metrics['accuracy']),
                'CV Std': metrics.get('cv_std', 0),
                'Train Time': metrics['train_time']
            })
    
    # Sort by test accuracy
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('Test Accuracy', ascending=False)
    
    print("\nTop 10 Models by Test Accuracy:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Model':<25} {'Category':<15} {'Accuracy':<12} {'CV Mean±Std':<15} {'Time(s)':<10}")
    print("-" * 80)
    
    for i, row in summary_df.head(10).iterrows():
        cv_str = f"{row['CV Mean']:.3f}±{row['CV Std']:.3f}" if row['CV Std'] > 0 else f"{row['CV Mean']:.3f}"
        print(f"{i+1:<5} {row['Model']:<25} {row['Category']:<15} {row['Test Accuracy']:.3f} {cv_str:<15} {row['Train Time']:.2f}")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    best_model = summary_df.iloc[0]
    print(f"\n1. BEST MODEL: {best_model['Model']} ({best_model['Test Accuracy']:.1%} accuracy)")
    
    # Find best in each category
    print("\n2. BEST PER CATEGORY:")
    for category in summary_df['Category'].unique():
        cat_best = summary_df[summary_df['Category'] == category].iloc[0]
        print(f"   - {category}: {cat_best['Model']} ({cat_best['Test Accuracy']:.1%})")
    
    # Speed vs accuracy trade-off
    fast_accurate = summary_df[summary_df['Train Time'] < 5].iloc[0]
    print(f"\n3. FASTEST ACCURATE: {fast_accurate['Model']} ({fast_accurate['Test Accuracy']:.1%} in {fast_accurate['Train Time']:.1f}s)")
    
    return summary_df


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_multiple_models.py <directory_path> [--features <type>]")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    feature_type = 'rb5_rel_diff'  # Default
    
    if len(sys.argv) > 3 and sys.argv[2] == '--features':
        feature_type = sys.argv[3]
    
    # Load data
    X, y, feature_names = load_kabsch_features(directory_path, feature_type)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale data for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test all model categories
    all_results = {}
    
    print("\nTesting models... This may take a few minutes.\n")
    
    all_results['SVM'] = test_svm_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    all_results['Tree'] = test_tree_models(X_train, X_test, y_train, y_test)
    all_results['Neural Network'] = test_neural_networks(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    all_results['Other'] = test_other_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    all_results['Ensemble'] = test_ensemble_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
    # Summarize all results
    summary_df = summarize_results(all_results)
    
    # Save results
    output_file = Path(directory_path) / 'model_comparison_results.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main() 