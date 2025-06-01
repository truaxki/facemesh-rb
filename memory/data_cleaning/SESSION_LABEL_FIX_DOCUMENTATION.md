# Session Label Fix Documentation

## Problem Description

**Date:** December 2024  
**Location:** `read/e26/` directory  
**Issue:** All CSV files in the folder had incorrect session labels in the "Test Name" column.

### The Issue
- **Expected format:** `session6`, `session7`, etc.
- **Actual format:** `e26 session6`, `e26 session7`, etc.
- **Affected files:** All session CSV files (e26-session1.csv through e26-session8.csv)
- **Total affected rows:** 866 rows across 8 files

### Files Affected
```
read/e26/
├── e26-baseline.csv (no changes needed - contained "baseline")
├── e26-session1.csv (106 rows fixed)
├── e26-session2.csv (104 rows fixed)
├── e26-session3.csv (107 rows fixed)
├── e26-session4.csv (106 rows fixed)
├── e26-session5.csv (118 rows fixed)
├── e26-session6.csv (106 rows fixed)
├── e26-session7.csv (112 rows fixed)
└── e26-session8.csv (107 rows fixed)
```

## Solution Implemented

### Script Created: `fix_session_labels.py`

A Python script was created to automatically process all CSV files and fix the session labels.

#### Key Features:
- **Automatic file discovery** - Finds all CSV files in the current directory
- **Safe processing** - Reads, modifies, and saves files without data loss
- **Pattern replacement** - Uses pandas string replacement to change "e26 session" to "session"
- **Progress reporting** - Shows detailed feedback for each file processed
- **Error handling** - Gracefully handles potential issues with file processing

#### How It Works:
1. Scans current directory for all `*.csv` files
2. Loads each CSV using pandas
3. Checks for "Test Name" column existence
4. Counts rows that need fixing (contain "e26 session")
5. Performs string replacement: `"e26 session" → "session"`
6. Saves the modified data back to the original file
7. Reports processing results

### Code Structure
```python
import os
import pandas as pd
import glob

def fix_session_labels():
    # Main processing function
    # - File discovery
    # - Data processing
    # - String replacement
    # - File saving
    # - Progress reporting

if __name__ == "__main__":
    # Script execution with directory management
```

## Usage Instructions

### Running the Script
```bash
# Navigate to the directory containing the CSV files
cd read/e26

# Execute the script
python fix_session_labels.py
```

### Expected Output
```
Working directory: C:\Users\...\facemesh-rb\read\e26
Found 9 CSV files to process.

Processing: e26-baseline.csv
  - No changes needed for e26-baseline.csv

Processing: e26-session1.csv
  ✓ Fixed 106 rows in e26-session1.csv

Processing: e26-session2.csv
  ✓ Fixed 104 rows in e26-session2.csv

[... continues for all files ...]

Processing complete!
```

## Results

### Before Fix
```csv
Subject Name,Test Name,Time (s),...
e26,e26 session6,0.0,...
e26,e26 session6,1.0,...
```

### After Fix
```csv
Subject Name,Test Name,Time (s),...
e26,session6,0.0,...
e26,session6,1.0,...
```

### Summary Statistics
- **Total files processed:** 9 CSV files
- **Files modified:** 8 session files
- **Files unchanged:** 1 baseline file (correctly formatted)
- **Total rows fixed:** 866 rows
- **Processing time:** < 30 seconds
- **Data integrity:** 100% preserved (no data loss)

## Technical Details

### Dependencies
- **Python 3.x**
- **pandas** - For CSV file reading/writing and data manipulation
- **glob** - For file pattern matching
- **os** - For directory operations

### Data Structure
The CSV files contain facemesh data with the following key columns:
- `Subject Name`: Always "e26"
- `Test Name`: The column that needed fixing
- `Time (s)`: Timestamp data
- Various feature columns (feat_0_x, feat_0_y, etc.)

### Safety Measures
- **In-place modification** - Files are updated directly (ensure backups exist)
- **Column validation** - Checks for "Test Name" column before processing
- **Error handling** - Continues processing other files if one fails
- **Pattern specificity** - Only replaces exact match "e26 session" pattern

## Future Considerations

### Prevention
- **Data validation scripts** during data collection
- **Automated testing** for label consistency
- **Standard naming conventions** documentation

### Reusability
The script can be easily adapted for similar label fixing tasks by:
- Modifying the search pattern (`'e26 session'`)
- Changing the replacement text (`'session'`)
- Adjusting the target column name (`'Test Name'`)

### Backup Recommendation
Always create backups before running data modification scripts:
```bash
# Create backup directory
mkdir backup
# Copy all CSV files to backup
cp *.csv backup/
```

---

**Script Location:** `read/e26/fix_session_labels.py`  
**Documentation Created:** December 2024  
**Status:** Completed Successfully ✅ 