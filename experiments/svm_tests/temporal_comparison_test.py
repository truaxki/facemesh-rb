#!/usr/bin/env python3
"""
TEMPORAL SEGMENTATION IMPACT ANALYSIS
=====================================
Compare SVM performance with and without temporal segmentation
on best and worst performing participants.

Best Performers: E17 (98.4%), E2 (96.5%), E18 (93.9%), E23 (92.2%)
Worst Performers: E6 (73.6%), E20 (78.7%), E3 (80.8%), E5 (81.5%)
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import os

def load_and_test_participant(participant_id):
    """Load participant data and test both temporal and non-temporal approaches"""
    
    # Load the dataset
    dataset_path = f"../../training/datasets/{participant_id}_all_sessions_expanded_kabsch.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
        
    print(f"\n🔬 TESTING PARTICIPANT: {participant_id.upper()}")
    print("=" * 60)
    
    df = pd.read_csv(dataset_path)
    
    # Extract features (exclude string columns and target)
    exclude_columns = ['target_session', 'participant_id', 'session_name']
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    X = df[feature_columns].values
    
    # Get original targets first
    y_temporal = df['target_session'].values
    
    # Handle NaN values - remove rows with NaN
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        print(f"   ⚠️  Found {nan_mask.sum()} rows with NaN values, removing them...")
        X = X[~nan_mask]
        y_temporal = y_temporal[~nan_mask]
    
    # Create non-temporal targets (remove _seg1, _seg2, etc.)
    y_non_temporal = []
    for target in y_temporal:
        # Remove temporal segment suffixes
        base_session = target.split('_seg')[0]  # Remove _seg1, _seg2, etc.
        y_non_temporal.append(base_session)
    
    y_non_temporal = np.array(y_non_temporal)
    
    print(f"📊 Dataset Info:")
    print(f"   • Total Frames: {len(X)}")
    print(f"   • Features: {len(feature_columns)}")
    print(f"   • Temporal Targets: {len(np.unique(y_temporal))} unique")
    print(f"   • Non-Temporal Targets: {len(np.unique(y_non_temporal))} unique")
    
    # Test both approaches
    results = {}
    
    # 1. WITH temporal segmentation (current approach)
    print(f"\n🎯 WITH Temporal Segmentation:")
    scaler_temporal = StandardScaler()
    X_scaled_temporal = scaler_temporal.fit_transform(X)
    
    svm_temporal = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    scores_temporal = cross_val_score(svm_temporal, X_scaled_temporal, y_temporal, cv=5, scoring='accuracy')
    
    mean_temporal = scores_temporal.mean()
    std_temporal = scores_temporal.std()
    
    print(f"   • Accuracy: {mean_temporal:.3f} ± {std_temporal:.3f}")
    print(f"   • Targets: {', '.join(np.unique(y_temporal)[:5])}{'...' if len(np.unique(y_temporal)) > 5 else ''}")
    
    results['temporal'] = {
        'mean': mean_temporal,
        'std': std_temporal,
        'targets': len(np.unique(y_temporal))
    }
    
    # 2. WITHOUT temporal segmentation
    print(f"\n🚫 WITHOUT Temporal Segmentation:")
    scaler_non_temporal = StandardScaler()
    X_scaled_non_temporal = scaler_non_temporal.fit_transform(X)
    
    svm_non_temporal = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    scores_non_temporal = cross_val_score(svm_non_temporal, X_scaled_non_temporal, y_non_temporal, cv=5, scoring='accuracy')
    
    mean_non_temporal = scores_non_temporal.mean()
    std_non_temporal = scores_non_temporal.std()
    
    print(f"   • Accuracy: {mean_non_temporal:.3f} ± {std_non_temporal:.3f}")
    print(f"   • Targets: {', '.join(np.unique(y_non_temporal)[:5])}{'...' if len(np.unique(y_non_temporal)) > 5 else ''}")
    
    results['non_temporal'] = {
        'mean': mean_non_temporal,
        'std': std_non_temporal,
        'targets': len(np.unique(y_non_temporal))
    }
    
    # Calculate impact
    impact = mean_temporal - mean_non_temporal
    impact_percent = (impact / mean_non_temporal) * 100 if mean_non_temporal > 0 else 0
    
    print(f"\n📈 TEMPORAL IMPACT:")
    print(f"   • Accuracy Gain: {impact:.3f} ({impact_percent:+.1f}%)")
    print(f"   • Target Increase: {results['temporal']['targets'] - results['non_temporal']['targets']} more classes")
    
    return results

def main():
    """Compare temporal vs non-temporal performance"""
    
    print("🧪 TEMPORAL SEGMENTATION IMPACT ANALYSIS")
    print("=" * 80)
    print("Testing impact of temporal segmentation on SVM performance")
    print("Comparing best vs worst performing participants")
    
    # Test participants
    test_participants = {
        'Best Performers': ['e17', 'e2', 'e18', 'e23'],
        'Worst Performers': ['e6', 'e20', 'e3', 'e5']
    }
    
    all_results = {}
    
    for category, participants in test_participants.items():
        print(f"\n\n🏆 {category.upper()}")
        print("=" * 80)
        
        category_results = {}
        
        for participant in participants:
            result = load_and_test_participant(participant)
            if result:
                category_results[participant] = result
        
        all_results[category] = category_results
    
    # Summary analysis
    print(f"\n\n📊 SUMMARY ANALYSIS")
    print("=" * 80)
    
    for category, results in all_results.items():
        print(f"\n{category}:")
        print("-" * 40)
        
        temporal_gains = []
        
        for participant, data in results.items():
            temporal_acc = data['temporal']['mean']
            non_temporal_acc = data['non_temporal']['mean']
            gain = temporal_acc - non_temporal_acc
            gain_percent = (gain / non_temporal_acc) * 100 if non_temporal_acc > 0 else 0
            
            temporal_gains.append(gain_percent)
            
            print(f"{participant.upper():4}: {non_temporal_acc:.3f} → {temporal_acc:.3f} ({gain_percent:+.1f}%)")
        
        if temporal_gains:
            avg_gain = np.mean(temporal_gains)
            print(f"     Average Temporal Gain: {avg_gain:+.1f}%")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print("   • Temporal segmentation creates more granular targets")
    print("   • More targets can improve accuracy (finer classification)")
    print("   • But may also lead to overfitting on temporal patterns")
    print("   • Real-world application needs careful target design")

if __name__ == "__main__":
    main() 