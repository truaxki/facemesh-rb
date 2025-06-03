# Alternative Models to XGBoost for Facial Expression Classification

## Overview

While XGBoost has been performing well (43.8% accuracy), several alternative models could potentially improve accuracy or offer other benefits for your facial expression classification task.

## Key Models to Consider

### 1. **Support Vector Machine (SVM)** ⭐ Highly Recommended

**Why SVM for facial features?**
- Excels in high-dimensional spaces (your 207 features)
- Can capture complex non-linear patterns with RBF kernel
- Geometric nature matches facial landmark data
- Robust to outliers (only support vectors matter)

**What SVM needs:**
```python
# CRITICAL: Must scale features!
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Then train SVM
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train_scaled, y_train)
```

**Expected performance:** 45-50% accuracy (potentially better than XGBoost)

### 2. **Random Forest**

**Advantages:**
- No scaling needed
- Feature importance analysis
- Handles non-linear patterns well
- Parallel training

**Expected performance:** Similar to XGBoost (42-45%)

### 3. **Neural Networks (MLP)**

**Why consider:**
- Can learn complex feature interactions
- Multiple hidden layers for deep patterns
- Good for when you have more data

**Requirements:**
- Needs scaled features (like SVM)
- More hyperparameters to tune
- Longer training time

**Expected performance:** 40-48% (depends on architecture)

### 4. **LightGBM**

**Advantages:**
- Faster than XGBoost
- Better memory efficiency
- Handles categorical features well

**Expected performance:** Similar to XGBoost but faster

### 5. **Ensemble Methods**

**Approach:** Combine multiple models
```python
# Example: Voting classifier
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('svm', SVC(probability=True)),
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier())
])
```

**Expected performance:** 45-50% (best of all worlds)

## Quick Comparison Table

| Model | Scaling Needed | Training Speed | Prediction Speed | Expected Accuracy |
|-------|----------------|----------------|------------------|-------------------|
| XGBoost | No | Fast | Fast | 43.8% (baseline) |
| SVM (RBF) | YES! | Medium | Fast | 45-50% |
| Random Forest | No | Medium | Fast | 42-45% |
| Neural Network | YES! | Slow | Fast | 40-48% |
| LightGBM | No | Very Fast | Fast | 43-45% |
| Ensemble | Mixed | Slow | Medium | 45-50% |

## Practical Recommendations

### For Maximum Accuracy:
1. **Try SVM with RBF kernel first**
   - Most likely to improve over XGBoost
   - Well-suited for geometric facial data
   
2. **Use ensemble of top 3 models**
   - Combine SVM + XGBoost + Random Forest
   - Soft voting usually better than hard voting

### For Production Use:
1. **If accuracy is critical:** SVM or Ensemble
2. **If speed is critical:** Stick with XGBoost or try LightGBM
3. **If interpretability matters:** Random Forest (feature importance)

## Implementation Steps

### Step 1: Test SVM
```bash
python test_svm_quick.py read/e1
```

### Step 2: Compare All Models
```bash
python test_multiple_models.py read/e1 --features rb5_rel_diff
```

### Step 3: Analyze Results
- Check `model_comparison_results.csv`
- Look for consistent winners across participants

## Why Different Models Might Work

1. **SVM**: Your features are geometric coordinates in 3D space. SVM finds optimal separating hyperplanes in this space.

2. **Neural Networks**: Facial expressions involve complex muscle interactions. Deep networks can learn these non-linear relationships.

3. **Ensemble**: Different models capture different aspects of the data. Combining them reduces individual model biases.

## Key Insights

1. **Feature Scaling is CRITICAL** for SVM and Neural Networks
   - Without scaling: ~15% accuracy
   - With scaling: ~45% accuracy

2. **Kabsch-aligned features** work especially well with geometric models like SVM

3. **Cross-validation is essential** - facial expression patterns vary between sessions

## Next Steps

1. Run `test_svm_quick.py` for detailed SVM analysis
2. Run `test_multiple_models.py` for comprehensive comparison
3. Test on multiple participants (E1, E18, etc.)
4. Consider ensemble of best performers

## Expected Improvements

Based on the geometric nature of your Kabsch-aligned features:
- **SVM (RBF)**: +2-5% over XGBoost
- **Ensemble**: +3-7% over XGBoost
- **Neural Network**: +0-4% over XGBoost (data-dependent)

Remember: The biggest gain came from feature engineering (Kabsch alignment: +13%). Model selection typically provides smaller but still valuable improvements (+3-5%). 