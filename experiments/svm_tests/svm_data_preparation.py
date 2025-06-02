#!/usr/bin/env python3
"""
SVM Data Preparation Pipeline
Builds on existing Kabsch-aligned features for optimal SVM performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys
import warnings
warnings.filterwarnings('ignore')

class SVMDataPreparator:
    """
    Advanced data preparation specifically optimized for SVM on facial expression data.
    """
    
    def __init__(self, feature_type='rb5_rel_mag', landmark_selection='nose_cheeks'):
        """
        Initialize SVM data preparator.
        
        Args:
            feature_type: 'rb5_rel_mag', 'rb5_rel_diff', or 'rb5_diff'
            landmark_selection: 'nose_cheeks', 'lips_eyebrows', or 'all_expression'
        """
        self.feature_type = feature_type
        self.landmark_selection = landmark_selection
        self.scaler = None
        self.feature_selector = None
        self.dimensionality_reducer = None
        self.feature_names = None
        
    def get_landmark_features(self, df):
        """Extract features based on landmark selection."""
        if self.landmark_selection == 'nose_cheeks':
            landmarks = [1, 2, 98, 327, 205, 425]  # Magic 6
        elif self.landmark_selection == 'lips_eyebrows':
            # Import from facial_clusters if available
            landmarks = list(range(0, 50))  # Simplified for demo
        else:
            landmarks = list(range(0, 478))  # All landmarks
        
        feature_cols = []
        
        if self.feature_type == 'rb5_rel_mag':
            for landmark_idx in landmarks:
                col_name = f'feat_{landmark_idx}_rb5_rel_mag'
                if col_name in df.columns:
                    feature_cols.append(col_name)
                    
        elif self.feature_type == 'rb5_rel_diff':
            for landmark_idx in landmarks:
                for axis in ['x', 'y', 'z']:
                    col_name = f'feat_{landmark_idx}_{axis}_rb5_rel_diff'
                    if col_name in df.columns:
                        feature_cols.append(col_name)
                        
        elif self.feature_type == 'rb5_diff':
            for landmark_idx in landmarks:
                for axis in ['x', 'y', 'z']:
                    col_name = f'feat_{landmark_idx}_{axis}_rb5_diff'
                    if col_name in df.columns:
                        feature_cols.append(col_name)
        
        return feature_cols
    
    def analyze_feature_distributions(self, X, feature_names):
        """Analyze feature distributions for scaling method selection."""
        print("\n📊 FEATURE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Basic statistics
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        mins = np.min(X, axis=0)
        maxes = np.max(X, axis=0)
        
        print(f"Features analyzed: {X.shape[1]}")
        print(f"Mean range: [{np.min(means):.6f}, {np.max(means):.6f}]")
        print(f"Std range: [{np.min(stds):.6f}, {np.max(stds):.6f}]")
        print(f"Min range: [{np.min(mins):.6f}, {np.max(mins):.6f}]")
        print(f"Max range: [{np.min(maxes):.6f}, {np.max(maxes):.6f}]")
        
        # Scale differences (important for SVM)
        scale_ratios = maxes / (stds + 1e-10)  # Avoid division by zero
        print(f"Scale ratio range: [{np.min(scale_ratios):.2f}, {np.max(scale_ratios):.2f}]")
        
        # Outlier detection
        outlier_features = []
        for i, feature in enumerate(feature_names):
            q75, q25 = np.percentile(X[:, i], [75, 25])
            iqr = q75 - q25
            outlier_threshold = 3 * iqr
            outliers = np.sum((X[:, i] < q25 - outlier_threshold) | 
                             (X[:, i] > q75 + outlier_threshold))
            if outliers > 0.05 * len(X):  # More than 5% outliers
                outlier_features.append((feature, outliers))
        
        if outlier_features:
            print(f"\n⚠️  Features with >5% outliers: {len(outlier_features)}")
            for feature, count in outlier_features[:5]:
                print(f"  - {feature}: {count} outliers ({count/len(X)*100:.1f}%)")
        
        return {
            'needs_robust_scaling': len(outlier_features) > 0.1 * X.shape[1],
            'high_scale_variance': np.max(scale_ratios) / np.min(scale_ratios) > 100,
            'outlier_features': outlier_features
        }
    
    def prepare_scaling_strategies(self, X, y, analysis_results):
        """Test different scaling strategies for SVM."""
        print("\n🔧 TESTING SCALING STRATEGIES")
        print("="*60)
        
        scalers = {
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'PowerTransformer': PowerTransformer(method='yeo-johnson'),
        }
        
        # Add quantile-based scaling for facial data
        scalers['QuantileUniform'] = PowerTransformer(method='quantile', 
                                                     output_distribution='uniform')
        
        results = {}
        
        # Split data for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        for scaler_name, scaler in scalers.items():
            try:
                # Fit and transform
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Quick SVM test
                svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
                svm.fit(X_train_scaled, y_train)
                accuracy = svm.score(X_test_scaled, y_test)
                
                # Analyze scaled distribution
                train_mean = np.mean(X_train_scaled)
                train_std = np.std(X_train_scaled)
                scale_uniformity = np.std(np.std(X_train_scaled, axis=0))
                
                results[scaler_name] = {
                    'accuracy': accuracy,
                    'mean': train_mean,
                    'std': train_std,
                    'scale_uniformity': scale_uniformity,
                    'scaler': scaler
                }
                
                print(f"{scaler_name:20} Accuracy: {accuracy:.3f}, "
                      f"Scale uniformity: {scale_uniformity:.4f}")
                
            except Exception as e:
                print(f"{scaler_name:20} Failed: {str(e)}")
        
        # Select best scaler
        best_scaler_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.scaler = results[best_scaler_name]['scaler']
        
        print(f"\n🏆 Best scaler: {best_scaler_name}")
        print(f"   Accuracy: {results[best_scaler_name]['accuracy']:.3f}")
        
        return results
    
    def feature_selection_for_svm(self, X_scaled, y):
        """Apply SVM-specific feature selection."""
        print("\n🎯 SVM-SPECIFIC FEATURE SELECTION")
        print("="*60)
        
        methods = {}
        
        # 1. Statistical tests (fast)
        f_selector = SelectKBest(f_classif, k=min(50, X_scaled.shape[1]))
        X_f_selected = f_selector.fit_transform(X_scaled, y)
        methods['F-test'] = {
            'selector': f_selector,
            'n_features': X_f_selected.shape[1],
            'scores': f_selector.scores_
        }
        
        # 2. Mutual information (non-linear relationships)
        mi_selector = SelectKBest(mutual_info_classif, k=min(30, X_scaled.shape[1]))
        X_mi_selected = mi_selector.fit_transform(X_scaled, y)
        methods['Mutual Info'] = {
            'selector': mi_selector,
            'n_features': X_mi_selected.shape[1],
            'scores': mi_selector.scores_
        }
        
        # 3. RFE with linear SVM (SVM-specific)
        if X_scaled.shape[1] <= 100:  # Only for manageable sizes
            linear_svm = SVC(kernel='linear', C=1.0, random_state=42)
            rfe_selector = RFE(linear_svm, n_features_to_select=min(20, X_scaled.shape[1]))
            X_rfe_selected = rfe_selector.fit_transform(X_scaled, y)
            methods['RFE-SVM'] = {
                'selector': rfe_selector,
                'n_features': X_rfe_selected.shape[1],
                'ranking': rfe_selector.ranking_
            }
        
        # Test each method
        best_method = None
        best_score = 0
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        for method_name, method_info in methods.items():
            selector = method_info['selector']
            
            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)
            
            # Test with SVM
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            svm.fit(X_train_selected, y_train)
            accuracy = svm.score(X_test_selected, y_test)
            
            print(f"{method_name:15} Features: {method_info['n_features']:3d}, "
                  f"Accuracy: {accuracy:.3f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_method = method_name
                self.feature_selector = selector
        
        print(f"\n🏆 Best feature selection: {best_method}")
        print(f"   Features selected: {methods[best_method]['n_features']}")
        print(f"   Accuracy: {best_score:.3f}")
        
        return methods
    
    def dimensionality_reduction_analysis(self, X_scaled, y):
        """Analyze dimensionality reduction options."""
        print("\n📉 DIMENSIONALITY REDUCTION ANALYSIS")
        print("="*60)
        
        if X_scaled.shape[1] <= 10:
            print("Feature count is already low, skipping dimensionality reduction")
            return None
        
        methods = {}
        target_dims = [min(10, X_scaled.shape[1]//2), min(20, X_scaled.shape[1])]
        
        for n_components in target_dims:
            if n_components >= X_scaled.shape[1]:
                continue
                
            # PCA
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            variance_explained = np.sum(pca.explained_variance_ratio_)
            
            # Test SVM performance
            X_train, X_test, y_train, y_test = train_test_split(
                X_pca, y, test_size=0.2, random_state=42, stratify=y
            )
            
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            svm.fit(X_train, y_train)
            accuracy = svm.score(X_test, y_test)
            
            methods[f'PCA-{n_components}'] = {
                'transformer': pca,
                'variance_explained': variance_explained,
                'accuracy': accuracy,
                'n_components': n_components
            }
            
            print(f"PCA-{n_components:2d}: Variance: {variance_explained:.3f}, "
                  f"Accuracy: {accuracy:.3f}")
            
            # FastICA (for non-Gaussian features)
            if n_components <= 20:  # ICA can be unstable with many components
                try:
                    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
                    X_ica = ica.fit_transform(X_scaled)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_ica, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
                    svm.fit(X_train, y_train)
                    accuracy = svm.score(X_test, y_test)
                    
                    methods[f'ICA-{n_components}'] = {
                        'transformer': ica,
                        'accuracy': accuracy,
                        'n_components': n_components
                    }
                    
                    print(f"ICA-{n_components:2d}: Accuracy: {accuracy:.3f}")
                    
                except Exception as e:
                    print(f"ICA-{n_components:2d}: Failed ({str(e)})")
        
        # Select best method if it improves performance
        if methods:
            best_method = max(methods.keys(), key=lambda k: methods[k]['accuracy'])
            baseline_accuracy = methods[best_method]['accuracy']  # Would need baseline comparison
            
            print(f"\n🏆 Best dimensionality reduction: {best_method}")
            print(f"   Accuracy: {methods[best_method]['accuracy']:.3f}")
            
            # Only use if it maintains >95% of original performance
            # self.dimensionality_reducer = methods[best_method]['transformer']
        
        return methods
    
    def create_svm_pipeline(self, X, y):
        """Create complete SVM preprocessing pipeline."""
        print("\n🔨 CREATING SVM PIPELINE")
        print("="*60)
        
        # Get feature names
        self.feature_names = self.get_landmark_features(X) if hasattr(X, 'columns') else None
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X_values = X[self.feature_names].values
        else:
            X_values = X
        
        # 1. Analyze distributions
        analysis = self.analyze_feature_distributions(X_values, self.feature_names)
        
        # 2. Select and fit scaler
        scaling_results = self.prepare_scaling_strategies(X_values, y, analysis)
        
        # 3. Scale the data
        X_scaled = self.scaler.fit_transform(X_values)
        
        # 4. Feature selection
        if X_scaled.shape[1] > 10:
            feature_results = self.feature_selection_for_svm(X_scaled, y)
            X_scaled = self.feature_selector.transform(X_scaled)
        
        # 5. Dimensionality reduction (if needed)
        if X_scaled.shape[1] > 20:
            dim_results = self.dimensionality_reduction_analysis(X_scaled, y)
        
        # Create pipeline
        pipeline_steps = [('scaler', self.scaler)]
        
        if self.feature_selector:
            pipeline_steps.append(('feature_selector', self.feature_selector))
            
        if self.dimensionality_reducer:
            pipeline_steps.append(('dim_reducer', self.dimensionality_reducer))
        
        pipeline = Pipeline(pipeline_steps)
        
        print(f"\n✅ SVM Pipeline created with {len(pipeline_steps)} steps")
        print(f"   Final feature count: {X_scaled.shape[1]}")
        
        return pipeline, X_scaled
    
    def analyze_class_balance(self, y):
        """Analyze class distribution for SVM class weighting."""
        print("\n⚖️  CLASS BALANCE ANALYSIS")
        print("="*60)
        
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        print(f"Classes: {len(unique)}")
        print(f"Class distribution:")
        
        min_count = np.min(counts)
        max_count = np.max(counts)
        balance_ratio = max_count / min_count
        
        for cls, count in zip(unique, counts):
            percentage = count / total * 100
            print(f"  {cls}: {count:4d} samples ({percentage:5.1f}%)")
        
        print(f"\nBalance ratio: {balance_ratio:.2f}:1")
        
        recommendations = []
        
        if balance_ratio > 3:
            recommendations.append("Use class_weight='balanced' in SVM")
            recommendations.append("Consider SMOTE for oversampling")
            
        if min_count < 30:
            recommendations.append("Small classes detected - use stratified CV")
            
        if len(unique) > 10:
            recommendations.append("Many classes - consider hierarchical classification")
        
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        return {
            'n_classes': len(unique),
            'balance_ratio': balance_ratio,
            'min_samples': min_count,
            'recommendations': recommendations
        }

def load_participant_data(participant_dir, feature_type='rb5_rel_mag'):
    """Load and combine participant data files."""
    participant_path = Path(participant_dir)
    
    # Find rb5-rel files
    rel_files = list(participant_path.glob('*-rb5-rel.csv'))
    
    if not rel_files:
        raise ValueError(f"No rb5-rel files found in {participant_dir}")
    
    all_data = []
    for file in rel_files:
        df = pd.read_csv(file)
        
        # Extract session type from filename
        parts = file.stem.split('-')
        if 'baseline' in parts:
            session_type = 'baseline'
        else:
            session_type = parts[1]  # session1, a, b, etc.
            
        df['session_type'] = session_type
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def main():
    """Demonstrate SVM data preparation pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python svm_data_preparation.py <participant_dir> [feature_type] [landmark_selection]")
        print("\nFeature types: rb5_rel_mag, rb5_rel_diff, rb5_diff")
        print("Landmark selections: nose_cheeks, lips_eyebrows, all_expression")
        sys.exit(1)
    
    participant_dir = sys.argv[1]
    feature_type = sys.argv[2] if len(sys.argv) > 2 else 'rb5_rel_mag'
    landmark_selection = sys.argv[3] if len(sys.argv) > 3 else 'nose_cheeks'
    
    print("🚀 SVM DATA PREPARATION PIPELINE")
    print("="*80)
    print(f"Participant: {participant_dir}")
    print(f"Feature type: {feature_type}")
    print(f"Landmark selection: {landmark_selection}")
    
    # Load data
    df = load_participant_data(participant_dir, feature_type)
    print(f"\nLoaded {len(df)} samples from {participant_dir}")
    
    # Initialize preparator
    preparator = SVMDataPreparator(feature_type, landmark_selection)
    
    # Get features
    feature_cols = preparator.get_landmark_features(df)
    if not feature_cols:
        print(f"❌ No features found for {feature_type} with {landmark_selection}")
        return
    
    X = df[feature_cols]
    y = df['session_type']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Analyze class balance
    balance_info = preparator.analyze_class_balance(y)
    
    # Create pipeline
    pipeline, X_processed = preparator.create_svm_pipeline(X, y)
    
    print(f"\n🎯 FINAL RECOMMENDATIONS FOR SVM")
    print("="*80)
    print(f"✅ Data is ready for SVM training")
    print(f"✅ {X_processed.shape[1]} features after preprocessing")
    print(f"✅ Pipeline includes optimal scaling for SVM")
    
    if balance_info['balance_ratio'] > 3:
        print(f"⚠️  Consider class_weight='balanced' due to imbalance")
    
    print(f"\nSuggested SVM parameters:")
    print(f"  - kernel='rbf' (good for facial expression patterns)")
    print(f"  - C=1.0 to 100.0 (start with 1.0)")
    print(f"  - gamma='scale' (automatic scaling)")
    
    if balance_info['balance_ratio'] > 3:
        print(f"  - class_weight='balanced' (due to imbalance)")
    
    print(f"\nNext steps:")
    print(f"  1. Use this pipeline for train/test splits")
    print(f"  2. Run hyperparameter optimization")
    print(f"  3. Test different kernels (RBF, polynomial, linear)")
    print(f"  4. Validate with cross-validation")

if __name__ == "__main__":
    main() 