# SYSTEM_PROMPT.md: Repository Navigation & Usage Guide

This document provides comprehensive guidance for effectively navigating and utilizing the `facemesh-rb` repository's organized structure.

## 🎯 **Repository Purpose & Context**

**facemesh-rb** is an advanced facial expression classification system that achieved a **96.8% accuracy breakthrough** through:
- **Kabsch alignment** for position-invariant features
- **SVM with proper scaling** (StandardScaler critical)
- **Ultra-efficient landmark discovery** (6 nose+cheek features = 25% accuracy)
- **Professional ML pipeline** with clean separation of concerns

## 📁 **Directory Structure & Navigation Strategy**

### **🎯 training/ - Core ML Pipeline**
**Purpose**: Production-ready machine learning pipeline with professional organization

```
training/
├── scripts/
│   ├── core/              # 🔧 FUNDAMENTAL UTILITIES
│   ├── features/          # 🧬 FEATURE ENGINEERING  
│   ├── analysis/          # 📊 RESEARCH & TESTING
│   ├── training/          # 🤖 MODEL TRAINING
│   ├── validation/        # ✅ DATA VALIDATION
│   └── demos/             # 🎯 TUTORIALS & DEMOS
├── datasets/              # 📁 PROCESSED DATA
└── models*/               # 💾 TRAINED OUTPUTS
```

#### **Navigation Rules for training/scripts/:**

**When working with core algorithms:**
```bash
cd training/scripts/core/
# - data_filters.py: Kabsch alignment implementation
# - facial_clusters.py: Landmark definitions (Magic 6, Expression regions)
# - data_loader_template.py: Data loading utilities
```

**When engineering features:**
```bash
cd training/scripts/features/
# - compute_rolling_baseline_with_kabsch.py: BREAKTHROUGH position-invariant features
# - compute_rolling_baseline_optimized.py: Performance-optimized version
# - compute_rolling_baseline.py: Original implementation
```

**When analyzing performance/efficiency:**
```bash
cd training/scripts/analysis/
# - test_kabsch_efficiency.py: Landmark efficiency testing
# - analyze_kabsch_features.py: Before/after Kabsch comparison
# - test_kabsch_lips_eyebrows.py: Expression region analysis
```

**When training models:**
```bash
cd training/scripts/training/
# - train_*.py: Various model training approaches
# - Focus on XGBoost implementations (pre-SVM breakthrough)
```

### **🧪 experiments/ - Research & Development**
**Purpose**: Experimental work, SVM research, and cutting-edge development

```
experiments/
└── svm_tests/            # 🎯 SVM MASTERY & SCALING
    ├── SVM_SCALING_EXPLAINED.md                    # 📚 BEGINNER TUTORIAL
    ├── FEATURE_SCALING_CRITICAL_EXPLANATION.md     # 🔬 DEEP TECHNICAL  
    ├── test_svm_*.py                              # 🧪 SVM EXPERIMENTS
    └── svm_data_preparation.py                    # 🔧 AUTOMATED PIPELINE
```

#### **experiments/ Usage Guidelines:**

**For SVM learning & development:**
```bash
cd experiments/svm_tests/

# STEP 1: Learn scaling fundamentals
cat SVM_SCALING_EXPLAINED.md

# STEP 2: Run experiments  
python test_svm_quick.py ../read/e1

# STEP 3: Use automated pipeline
python svm_data_preparation.py ../read/e1 rb5_rel_mag nose_cheeks
```

**Key SVM Insights Stored Here:**
- **Feature scaling is NON-NEGOTIABLE** (40.5% → 96.8% accuracy)
- **StandardScaler implementation** (mean=0, std=1 transformation)
- **Production-ready SVM pipeline** with automated preprocessing
- **Hyperparameter optimization strategies**

### **📚 memory/ - Research Findings & Documentation**
**Purpose**: Historical research findings, breakthrough documentation, and institutional knowledge

#### **Types of Information in memory/:**

**Breakthrough Documentation:**
- **Kabsch alignment discoveries** and mathematical foundations
- **XGBoost optimization results** and hyperparameter findings  
- **Alternative model exploration** (Random Forest, Neural Networks, etc.)
- **Feature scaling mathematical explanations**

**Research Methodology:**
- **Performance comparison matrices**
- **Failed experiment documentation** (what NOT to do)
- **Evolution of understanding** (how we got to 96.8% accuracy)
- **Cross-validation strategies** and temporal considerations

**Usage Pattern for memory/:**
```bash
# Reference historical findings
cd memory/
ls *.md  # Browse available research documents

# Understand evolution of discoveries
cat KABSCH_SVM_COMPLETE_ANALYSIS.md
cat ALTERNATIVE_MODELS_GUIDE.md
```

### **📖 docs/ - Documentation & Guides**
**Purpose**: Comprehensive documentation, research frameworks, and implementation guides

**Contents:**
- **`SVM_OPTIMIZATION_RESEARCH.md`**: Complete SVM research framework
- **Implementation roadmaps** and performance targets
- **Architecture documentation** and design decisions

### **📥 read/ - Raw Participant Data**
**Purpose**: Unprocessed participant data (e1, e2, ..., e30)

**Structure per participant:**
```
read/e1/
├── baseline.csv, session1.csv, session2.csv, ...
├── a.csv, b.csv, c.csv, ...  # Experimental conditions
└── *-rb5-rel.csv            # Kabsch-aligned features (generated)
```

## 🧠 **Key System Knowledge & Workflows**

### **The 96.8% Accuracy Formula:**
```python
# CRITICAL SUCCESS PIPELINE:
Raw_Landmarks → Z_Scaling → Rolling_Baseline → Kabsch_Alignment → Feature_Scaling → SVM → 96.8%

# WITHOUT feature scaling: ~40% accuracy  
# WITH proper scaling: ~97% accuracy
```

### **Landmark Efficiency Hierarchy:**
1. **Magic 6 (Nose+Cheeks)**: 4.21% efficiency per feature (25% total accuracy)
2. **Lips+Eyebrows**: 0.63% efficiency per feature (43.8% total accuracy)  
3. **All Expression**: 0.47% efficiency per feature (96.8% total accuracy)

### **Feature Type Decision Tree:**
```
Use Case?
├── Real-time/IoT → rb5_rel_mag + nose_cheeks (6 features)
├── Production → rb5_rel_diff + selected features (20-30 features)  
├── Research → rb5_rel_diff + lips_eyebrows (69-207 features)
└── Maximum → rb5_rel_diff + all_expression (207 features)
```

### **Algorithm Selection Guide:**
```
Data Type?
├── Position-invariant (Kabsch) → SVM (97% accuracy) 
├── Raw geometric data → Linear SVM (95.5% accuracy)
├── Mixed/complex patterns → XGBoost (44% accuracy)
└── Legacy/comparison → Random Forest (42% accuracy)
```

## 🔧 **Effective Usage Patterns**

### **For New Research Questions:**
1. **Start in experiments/**: Develop new approaches in isolated environment
2. **Reference memory/**: Check what's been tried before
3. **Use training/scripts/analysis/**: Test on established datasets
4. **Validate in training/scripts/**: Integrate into production pipeline

### **For SVM Development:**
1. **experiments/svm_tests/SVM_SCALING_EXPLAINED.md**: Learn fundamentals
2. **experiments/svm_tests/test_svm_*.py**: Experiment with approaches
3. **experiments/svm_tests/svm_data_preparation.py**: Production pipeline
4. **training/scripts/features/**: Generate Kabsch-aligned features

### **For Feature Engineering:**
1. **training/scripts/core/data_filters.py**: Understand Kabsch implementation
2. **training/scripts/features/**: Generate position-invariant features
3. **training/scripts/analysis/**: Test feature effectiveness
4. **experiments/svm_tests/**: Optimize for SVM performance

### **For Performance Analysis:**
1. **training/scripts/analysis/test_kabsch_efficiency.py**: Landmark efficiency
2. **training/scripts/analysis/analyze_kabsch_features.py**: Before/after comparison
3. **experiments/svm_tests/**: SVM-specific performance testing
4. **memory/**: Historical performance context

## ⚠️ **Critical Knowledge & Gotchas**

### **SVM Scaling is NON-NEGOTIABLE:**
```python
# THIS WILL FAIL (~40% accuracy):
svm = SVC()
svm.fit(X_raw, y)

# THIS SUCCEEDS (~97% accuracy):
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
svm = SVC()
svm.fit(X_scaled, y)
```

### **Kabsch Alignment Prerequisites:**
- **Z-scaling required**: `z_scaled = z_raw * face_depth_cm`
- **Temporal ordering**: Frames must be chronologically sorted
- **Baseline computation**: Need minimum 5 frames for rb5 features

### **File Path Navigation:**
```bash
# From experiments/svm_tests/ to participant data:
python test_svm_quick.py ../read/e1

# From training/scripts/analysis/ to participant data:  
python test_kabsch_efficiency.py ../../read/e1

# From training/scripts/features/ to datasets:
python compute_rolling_baseline_with_kabsch.py ../datasets/
```

### **Import Path Updates:**
Due to reorganization, update imports:
```python
# OLD (before organization):
from data_filters import kabsch_align
from facial_clusters import FACIAL_CLUSTERS

# NEW (after organization):
from core.data_filters import kabsch_align  
from core.facial_clusters import FACIAL_CLUSTERS
```

## 🎯 **Workflow Templates**

### **New SVM Experiment:**
```bash
# 1. Learn scaling concepts
cd experiments/svm_tests/
cat SVM_SCALING_EXPLAINED.md

# 2. Test on participant data
python test_svm_quick.py ../read/e1 rb5_rel_mag nose_cheeks

# 3. Analyze results  
python svm_data_preparation.py ../read/e1

# 4. Document findings in memory/
```

### **Feature Engineering Workflow:**
```bash
# 1. Generate Kabsch features
cd training/scripts/features/
python compute_rolling_baseline_with_kabsch.py ../../read/e1

# 2. Test efficiency
cd ../analysis/
python test_kabsch_efficiency.py ../../read/e1  

# 3. Optimize for SVM
cd ../../experiments/svm_tests/
python svm_data_preparation.py ../read/e1
```

### **Performance Analysis:**
```bash
# 1. Baseline efficiency
cd training/scripts/analysis/
python test_kabsch_efficiency.py ../../read/e1

# 2. Cross-method comparison  
python analyze_kabsch_features.py ../../read/e1

# 3. SVM optimization
cd ../../experiments/svm_tests/
python test_svm_kabsch_comparison.py ../read/e1
```

## 📝 **Documentation Standards**

### **When Adding New Files:**
- **experiments/**: Experimental/research code with detailed comments
- **training/scripts/**: Production-quality code with error handling
- **memory/**: Research findings with methodology and results
- **docs/**: Comprehensive guides with examples

### **When Documenting Discoveries:**
1. **Record in memory/**: Historical context and methodology
2. **Update experiments/**: Practical implementation
3. **Integrate in training/**: Production-ready version  
4. **Reference in docs/**: Comprehensive documentation

---

## 🚀 **Quick Reference Commands**

```bash
# Learn SVM scaling fundamentals:
cd experiments/svm_tests && cat SVM_SCALING_EXPLAINED.md

# Test landmark efficiency:
cd training/scripts/analysis && python test_kabsch_efficiency.py ../../read/e1

# Generate Kabsch features:
cd training/scripts/features && python compute_rolling_baseline_with_kabsch.py ../../read/e1

# Run SVM experiments:
cd experiments/svm_tests && python test_svm_quick.py ../read/e1

# Automated SVM pipeline:
cd experiments/svm_tests && python svm_data_preparation.py ../read/e1 rb5_rel_mag nose_cheeks
```

---

**Remember**: This system achieved **96.8% accuracy** through proper Kabsch alignment + SVM scaling. The organization reflects this breakthrough - use `experiments/svm_tests/` for SVM mastery, `training/scripts/` for production workflows, and `memory/` for historical context. 