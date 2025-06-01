# Rolling Baseline Feature Documentation

**Date:** December 2024  
**Script:** `compute_rolling_baseline.py`  
**Purpose:** Compute temporal rolling average baseline features for facemesh coordinate data  

## Overview

The rolling baseline feature computation adds temporal context to facemesh data by calculating rolling averages and deviations for each coordinate over time. This helps identify patterns in facial movement stability and variation.

## Feature Description

### Rolling Baseline (RB) Features

For each coordinate feature (e.g., `feat_373_y`), the script computes:

1. **Rolling Average (`_rb{n}`)**: The mean of the last `n` time points (frames) including the current frame
   - Example: `feat_373_y_rb10` = average of `feat_373_y` over the last 10 frames
   
2. **Rolling Difference (`_rb{n}_diff`)**: The deviation from the rolling average
   - Example: `feat_373_y_rb10_diff` = `feat_373_y` - `feat_373_y_rb10`

### Window Sizes

Default windows: **5** and **10** frames
- `_rb5`: 5-frame rolling average (shorter-term baseline)
- `_rb10`: 10-frame rolling average (longer-term baseline)

## Implementation Details

### Input Processing
- **Input files**: Original cleaned CSV files (e.g., `e25-baseline.csv`)
- **Output files**: New files with `-rb` suffix (e.g., `e25-baseline-rb.csv`)
- **Preserved data**: All original columns are retained

### Features Added
For each of the 1,434 coordinate features (478 landmarks × 3 axes):
- 2 rolling averages (rb5, rb10)
- 2 difference features (rb5_diff, rb10_diff)
- **Total new features**: 5,736 columns per file

### Column Naming Convention
```
Original: feat_{landmark}_{axis}
RB Average: feat_{landmark}_{axis}_rb{window}
RB Difference: feat_{landmark}_{axis}_rb{window}_diff

Example:
- Original: feat_373_y
- 5-frame average: feat_373_y_rb5
- 5-frame difference: feat_373_y_rb5_diff
- 10-frame average: feat_373_y_rb10
- 10-frame difference: feat_373_y_rb10_diff
```

## Use Cases

### 1. Movement Stability Analysis
- **Low `_rb{n}_diff` values**: Indicate stable facial position
- **High `_rb{n}_diff` values**: Indicate rapid movement or tremor

### 2. Baseline Deviation Detection
- Compare current position to recent average
- Identify sudden changes or anomalies

### 3. Temporal Smoothing
- Use rolling averages to reduce noise in measurements
- Different window sizes capture different temporal scales

### 4. Feature Engineering for ML
- Provides temporal context for each time point
- Useful for models that benefit from movement dynamics

## Technical Specifications

### Rolling Window Parameters
- **Method**: Simple moving average
- **Min periods**: 1 (handles edge cases at start of sequence)
- **Alignment**: Right-aligned (includes current frame)

### Performance Considerations
- **File size increase**: ~5x original size
- **Processing time**: ~2-3 seconds per file
- **Memory usage**: Proportional to file size and column count

## Usage

### Process Single File
```bash
python compute_rolling_baseline.py path/to/file.csv
```

### Process Directory
```bash
python compute_rolling_baseline.py directory_path
```

### Custom Window Sizes
```bash
# Single file with custom windows
python compute_rolling_baseline.py file.csv 5,10,20

# Directory with custom windows
python compute_rolling_baseline.py . 3,7,15
```

## Output Statistics

### Example File Transformation
- **Original**: 1,439 columns, 1.1 MB
- **With RB features**: 7,175 columns, 5.6 MB
- **Processing time**: ~2.5 seconds

### Expected Results
For a typical dataset with:
- 30 subjects × 9 files = 270 files
- Excluding 7 empty files = 263 files to process
- Total processing time: ~10-15 minutes
- Total storage increase: ~1 GB additional space

## Data Quality Notes

### Edge Cases Handled
1. **Empty files**: Skipped with warning
2. **Start of sequence**: Uses available frames (min_periods=1)
3. **Missing data**: Propagated through rolling calculations

### Validation Checks
- Ensures only base feature columns are processed
- Skips derived features (diff, rb, tot)
- Preserves original data integrity

## Integration with Analysis Pipeline

### Recommended Workflow
1. Clean and standardize data (`validate_and_standardize_csvs.py`)
2. Add rolling baseline features (`compute_rolling_baseline.py`)
3. Perform analysis on enhanced dataset

### Downstream Applications
- Time series classification
- Anomaly detection
- Movement pattern analysis
- Stability metrics computation

---

**Status:** Feature computation script ready and tested ✅  
**Next Steps:** Complete processing of all files, then proceed with analysis using enhanced features 