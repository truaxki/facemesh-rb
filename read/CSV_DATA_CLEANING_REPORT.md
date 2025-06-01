# CSV Data Cleaning and Validation Report

**Date:** December 2024  
**Script:** `validate_and_standardize_csvs.py`  
**Location:** `/read/` directory and all subdirectories  

## Executive Summary

A comprehensive validation and standardization process was performed on 158 CSV files containing facemesh data across multiple subjects (e1-e30). The script identified and fixed numerous data consistency issues, standardized naming conventions, and ensured proper data sorting.

### Key Statistics
- **Total files processed:** 158 CSV files
- **Files with issues found:** 9 files
- **Files successfully fixed:** 29 files
- **Test Names corrected:** 26 instances
- **Subject Names corrected:** 3 instances
- **Empty files identified:** 7 files
- **Files sorted by time:** 54 files (in initial run)

## Data Cleaning Performed

### 1. Test Name Standardization

**Issue:** Several subjects had Test Names with subject prefixes (e.g., "e20 baseline" instead of "baseline")

**Affected Subjects:**
- **e20**: All 8 files (baseline + 7 sessions) - Fixed ✅
- **e22**: All 9 files (baseline + 8 sessions) - Fixed ✅
- **e25**: All 8 files (baseline + 7 sessions, excluding missing session5) - Fixed ✅

**Fix Applied:**
```
"e20 baseline" → "baseline"
"e20 session1" → "session1"
"e20 session2" → "session2"
... etc.
```

**Special Case - e26:**
- e26-baseline.csv had mixed Test Names: ['baseline', 'e26 baseline']
- Standardized all rows to use 'baseline' - Fixed ✅

### 2. Subject Name Corrections

**Issue:** All files in the e17 folder had Subject Name "e26" instead of "e17"

**Files Fixed:**
- e17/e17-baseline.csv: Subject Name 'e26' → 'e17' ✅
- e17/e17-session1.csv: Subject Name 'e26' → 'e17' ✅
- e17/e17-session2.csv: Subject Name 'e26' → 'e17' ✅

### 3. Time Series Sorting

**Issue:** Many files had records not properly sorted by Time (s)

**Action:** 54 files were automatically sorted in ascending order by Time (s) field during the initial validation run

### 4. Empty Files Identified

**Issue:** Several files contain no data rows (0 records)

**Empty Files Found:**
1. `e17/e17-session3.csv` - 0 rows ⚠️
2. `e17/e17-session4.csv` - 0 rows ⚠️
3. `e17/e17-session6.csv` - 0 rows ⚠️
4. `e17/e17-session7.csv` - 0 rows ⚠️
5. `e17/e17-session8.csv` - 0 rows ⚠️
6. `e20/e20-session2.csv` - 0 rows ⚠️
7. `e25/e25-session5.csv` - 0 rows ⚠️

**Note:** These empty files were preserved but flagged for investigation

## Data Structure Analysis

### Field Categories Identified

1. **Metadata Fields (5 fields)**
   - Subject Name
   - Test Name
   - Time (s)
   - Face Depth (cm)
   - Unnamed: 0

2. **Feature Coordinates (1,434 fields)**
   - 478 X coordinates (feat_0_x to feat_477_x)
   - 478 Y coordinates (feat_0_y to feat_477_y)
   - 478 Z coordinates (feat_0_z to feat_477_z)

3. **Difference Features (1,912 fields)**
   - *_xdiff, *_ydiff, *_zdiff features
   - *_ztot aggregate features

4. **Survey Data Fields (13 fields)**
   - Performance and stress-related questionnaire responses
   - Found only in: `self-reported-data(original-sorted).csv`

## Data Anomalies and Outliers

### 1. Missing Data Files
- **e4**: Missing session7.csv
- **e5**: Missing session4.csv
- **e25**: session5.csv exists but is empty

### 2. Low Record Count Files
- **e3/e3-session5.csv**: Only 1 row (typical files have 100+ rows)
- **e30/e30-session8.csv**: Only 25 rows (significantly fewer than normal)
- **e27/e27-session2.csv**: Only 56 rows (about half the typical amount)

### 3. Different Data Structure
- **self-reported-data(original-sorted).csv**: Contains questionnaire data instead of facemesh coordinates
  - Missing required columns: Subject Name, Test Name, Time (s)
  - Contains stress and performance assessment fields

## Validation Rules Applied

1. **Consistency Check**: Each file must have consistent Subject Name and Test Name across all rows
2. **Required Columns**: All facemesh files must contain: Subject Name, Test Name, Time (s)
3. **Naming Convention**: 
   - Subject Names must match folder name (e.g., files in e17 folder must have Subject Name "e17")
   - Test Names should be simple: "baseline", "session1", "session2", etc. (no subject prefixes)
4. **Sorting**: All records must be sorted by Time (s) in ascending order
5. **Duplicate Time Check**: Flag files with duplicate time values (informational only)

## Post-Cleaning Data State

### ✅ Successfully Standardized
- All Test Names now follow consistent format (no subject prefixes)
- All Subject Names match their folder location
- All files sorted by time
- Mixed Test Names in e26-baseline.csv resolved

### ⚠️ Remaining Issues
1. **Empty files** (7 files) - Require investigation for data collection issues
2. **Missing files** - e4/session7, e5/session4
3. **Low record count files** - May indicate incomplete data collection
4. **Non-facemesh file** - self-reported-data file has different structure

## Recommendations

1. **Investigate Empty Files**: Determine why 7 files have no data and whether re-collection is needed
2. **Review Low Record Files**: Check if e3-session5 (1 row) and e30-session8 (25 rows) have complete data
3. **Locate Missing Sessions**: Find or recreate missing session files for e4 and e5
4. **Separate Data Types**: Move questionnaire data (self-reported-data) to a different location or clearly mark as non-facemesh data
5. **Implement Validation**: Add data validation during collection to prevent naming inconsistencies
6. **Regular Audits**: Run validation script periodically to catch issues early

## Script Usage

### Validation Only (Read-Only Mode)
```bash
python validate_and_standardize_csvs.py [directory]
```

### Validation with Automatic Fixes
```bash
python validate_and_standardize_csvs.py [directory] --fix
```

### Output Files
- **Console Output**: Real-time processing status and summary
- **csv_validation_report.txt**: Detailed report of all findings and fixes applied

---

**Status:** Data cleaning completed successfully ✅  
**Next Steps:** Review empty files and low record count files for potential data collection issues 