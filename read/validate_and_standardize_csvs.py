import os
import pandas as pd
import glob
from pathlib import Path
from collections import defaultdict
import sys
import re
from datetime import datetime

def validate_and_standardize_csvs(root_directory=".", fix_issues=True):
    """
    Validate and standardize all CSV files in a directory and its subdirectories.
    
    - Ensures consistent Subject Name and Test Name within each file
    - Sorts records by Time (s) field
    - Reports all unique field names across all files
    - Identifies outliers and inconsistencies
    - Optionally fixes common issues
    """
    
    # Statistics tracking
    stats = {
        'files_processed': 0,
        'files_with_issues': 0,
        'files_sorted': 0,
        'files_fixed': 0,
        'test_names_fixed': 0,
        'subject_names_fixed': 0,
        'empty_files': [],
        'inconsistent_files': [],
        'fixes_applied': []
    }
    
    # Collect all unique field names
    all_field_names = set()
    
    # Track Subject Name and Test Name combinations
    file_metadata = {}
    
    # Find all CSV files recursively
    csv_files = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print(f"No CSV files found in {root_directory} or its subdirectories.")
        return stats
    
    print(f"Found {len(csv_files)} CSV files to process.\n")
    print("="*80)
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            print(f"\nProcessing: {csv_file}")
            stats['files_processed'] += 1
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Check if file is empty
            if len(df) == 0:
                print(f"  ⚠️  EMPTY FILE - No data rows")
                stats['empty_files'].append(csv_file)
                stats['files_with_issues'] += 1
                continue
            
            # Collect field names
            all_field_names.update(df.columns.tolist())
            
            # Check if required columns exist
            required_columns = ['Subject Name', 'Test Name', 'Time (s)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"  ⚠️  Missing required columns: {missing_columns}")
                stats['files_with_issues'] += 1
                stats['inconsistent_files'].append({
                    'file': csv_file,
                    'issue': f"Missing columns: {missing_columns}"
                })
                continue
            
            # Extract expected subject name from folder structure
            folder_path = os.path.dirname(csv_file)
            folder_name = os.path.basename(folder_path)
            expected_subject = folder_name  # e.g., "e17", "e26", etc.
            
            # Track if file was modified
            file_modified = False
            
            # Check and fix Subject Name consistency
            unique_subjects = df['Subject Name'].unique()
            
            if len(unique_subjects) > 1:
                print(f"  ❌ INCONSISTENT Subject Names found: {unique_subjects}")
                stats['files_with_issues'] += 1
                stats['inconsistent_files'].append({
                    'file': csv_file,
                    'issue': f"Multiple Subject Names: {unique_subjects}"
                })
            else:
                current_subject = unique_subjects[0]
                # Check if subject name matches folder name
                if current_subject != expected_subject and expected_subject.startswith('e') and expected_subject[1:].isdigit():
                    if fix_issues:
                        print(f"  🔧 Fixing Subject Name: '{current_subject}' → '{expected_subject}'")
                        df['Subject Name'] = expected_subject
                        file_modified = True
                        stats['subject_names_fixed'] += 1
                        stats['fixes_applied'].append({
                            'file': csv_file,
                            'fix': f"Subject Name: '{current_subject}' → '{expected_subject}'"
                        })
                    else:
                        print(f"  ❌ Subject Name mismatch: '{current_subject}' (expected '{expected_subject}')")
                else:
                    print(f"  ✓ Subject Name: '{current_subject}' (consistent)")
            
            # Check and fix Test Name consistency
            unique_tests = df['Test Name'].unique()
            
            if len(unique_tests) > 1:
                print(f"  ❌ INCONSISTENT Test Names found: {unique_tests}")
                stats['files_with_issues'] += 1
                stats['inconsistent_files'].append({
                    'file': csv_file,
                    'issue': f"Multiple Test Names: {unique_tests}"
                })
                
                # Special handling for e26 baseline with mixed names
                if fix_issues and 'baseline' in unique_tests and any(t.endswith('baseline') for t in unique_tests):
                    print(f"  🔧 Standardizing all Test Names to 'baseline'")
                    df['Test Name'] = 'baseline'
                    file_modified = True
                    stats['test_names_fixed'] += 1
                    stats['fixes_applied'].append({
                        'file': csv_file,
                        'fix': f"Test Names: {unique_tests} → 'baseline'"
                    })
            else:
                current_test = unique_tests[0]
                # Check if test name has subject prefix (e.g., "e20 session1")
                if ' ' in current_test and current_test.startswith(expected_subject + ' '):
                    if fix_issues:
                        new_test = current_test.replace(expected_subject + ' ', '')
                        print(f"  🔧 Fixing Test Name: '{current_test}' → '{new_test}'")
                        df['Test Name'] = new_test
                        file_modified = True
                        stats['test_names_fixed'] += 1
                        stats['fixes_applied'].append({
                            'file': csv_file,
                            'fix': f"Test Name: '{current_test}' → '{new_test}'"
                        })
                    else:
                        print(f"  ❌ Test Name has subject prefix: '{current_test}'")
                else:
                    print(f"  ✓ Test Name: '{current_test}' (consistent)")
            
            # Store metadata for this file
            file_metadata[csv_file] = {
                'subject_names': df['Subject Name'].unique().tolist() if file_modified else unique_subjects.tolist(),
                'test_names': df['Test Name'].unique().tolist() if file_modified else unique_tests.tolist(),
                'row_count': len(df)
            }
            
            # Check if Time (s) column needs sorting
            time_values = df['Time (s)'].values
            is_sorted = all(time_values[i] <= time_values[i+1] for i in range(len(time_values)-1))
            
            if not is_sorted:
                print(f"  🔄 Sorting by Time (s)...")
                df = df.sort_values('Time (s)', ascending=True)
                file_modified = True
                stats['files_sorted'] += 1
                print(f"  ✓ File sorted by Time (s)")
            else:
                print(f"  ✓ Already sorted by Time (s)")
            
            # Save file if modified
            if file_modified and fix_issues:
                df.to_csv(csv_file, index=False)
                stats['files_fixed'] += 1
                print(f"  💾 File saved with fixes")
            
            # Check for time duplicates
            time_duplicates = df[df.duplicated(subset=['Time (s)'], keep=False)]
            if not time_duplicates.empty:
                print(f"  ⚠️  Found {len(time_duplicates)} rows with duplicate Time (s) values")
                duplicate_times = time_duplicates['Time (s)'].unique()
                print(f"     Duplicate times: {duplicate_times[:5]}{'...' if len(duplicate_times) > 5 else ''}")
                
        except Exception as e:
            print(f"  ✗ Error processing {csv_file}: {str(e)}")
            stats['files_with_issues'] += 1
            stats['inconsistent_files'].append({
                'file': csv_file,
                'issue': f"Error: {str(e)}"
            })
    
    # Print summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    print(f"\nProcessing Statistics:")
    print(f"  Total files processed: {stats['files_processed']}")
    print(f"  Files with issues: {stats['files_with_issues']}")
    print(f"  Files fixed: {stats['files_fixed']}")
    print(f"  Files sorted: {stats['files_sorted']}")
    print(f"  Test Names fixed: {stats['test_names_fixed']}")
    print(f"  Subject Names fixed: {stats['subject_names_fixed']}")
    print(f"  Empty files found: {len(stats['empty_files'])}")
    
    print(f"\n📋 All Unique Field Names ({len(all_field_names)} total):")
    print("-"*80)
    
    # Group field names by category for better readability
    field_categories = {
        'metadata': [],
        'feature_x': [],
        'feature_y': [],
        'feature_z': [],
        'diff_features': [],
        'other': []
    }
    
    for field in sorted(all_field_names):
        if field in ['Subject Name', 'Test Name', 'Time (s)', 'Face Depth (cm)', 'Unnamed: 0']:
            field_categories['metadata'].append(field)
        elif field.endswith('_x') and not field.endswith('_xdiff'):
            field_categories['feature_x'].append(field)
        elif field.endswith('_y') and not field.endswith('_ydiff'):
            field_categories['feature_y'].append(field)
        elif field.endswith('_z') and not field.endswith('_zdiff') and not field.endswith('_ztot'):
            field_categories['feature_z'].append(field)
        elif field.endswith(('_xdiff', '_ydiff', '_zdiff', '_ztot')):
            field_categories['diff_features'].append(field)
        else:
            field_categories['other'].append(field)
    
    print("\nMetadata Fields:")
    for field in field_categories['metadata']:
        print(f"  • {field}")
    
    print(f"\nFeature X coordinates ({len(field_categories['feature_x'])} fields):")
    print(f"  {field_categories['feature_x'][:5]}{'...' if len(field_categories['feature_x']) > 5 else ''}")
    
    print(f"\nFeature Y coordinates ({len(field_categories['feature_y'])} fields):")
    print(f"  {field_categories['feature_y'][:5]}{'...' if len(field_categories['feature_y']) > 5 else ''}")
    
    print(f"\nFeature Z coordinates ({len(field_categories['feature_z'])} fields):")
    print(f"  {field_categories['feature_z'][:5]}{'...' if len(field_categories['feature_z']) > 5 else ''}")
    
    if field_categories['diff_features']:
        print(f"\nDifference Features ({len(field_categories['diff_features'])} fields):")
        print(f"  {field_categories['diff_features'][:5]}{'...' if len(field_categories['diff_features']) > 5 else ''}")
    
    if field_categories['other']:
        print(f"\nOther Fields:")
        for field in field_categories['other'][:10]:  # Show first 10
            print(f"  • {field}")
        if len(field_categories['other']) > 10:
            print(f"  ... and {len(field_categories['other']) - 10} more fields")
    
    # Report on Subject/Test Name patterns
    print("\n📊 Subject/Test Name Patterns:")
    print("-"*80)
    
    # Group files by subject and test name
    patterns = defaultdict(list)
    for file, metadata in file_metadata.items():
        key = (tuple(metadata['subject_names']), tuple(metadata['test_names']))
        patterns[key].append(file)
    
    for (subjects, tests), files in patterns.items():
        print(f"\nPattern: Subject={subjects}, Test={tests}")
        print(f"  Files with this pattern: {len(files)}")
        for file in files[:3]:  # Show first 3 files
            print(f"    - {os.path.relpath(file, root_directory)}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more files")
    
    # Report empty files
    if stats['empty_files']:
        print("\n📄 Empty Files (0 rows):")
        print("-"*80)
        for file in stats['empty_files']:
            print(f"  - {os.path.relpath(file, root_directory)}")
    
    # Report fixes applied
    if stats['fixes_applied']:
        print("\n🔧 Fixes Applied:")
        print("-"*80)
        for fix in stats['fixes_applied'][:10]:  # Show first 10
            print(f"\n  File: {os.path.relpath(fix['file'], root_directory)}")
            print(f"  Fix: {fix['fix']}")
        if len(stats['fixes_applied']) > 10:
            print(f"\n  ... and {len(stats['fixes_applied']) - 10} more fixes")
    
    # Report remaining issues
    remaining_issues = [issue for issue in stats['inconsistent_files'] 
                       if not any(fix['file'] == issue['file'] for fix in stats['fixes_applied'])]
    if remaining_issues:
        print("\n⚠️  Remaining Issues:")
        print("-"*80)
        for issue in remaining_issues:
            print(f"\n  File: {os.path.relpath(issue['file'], root_directory)}")
            print(f"  Issue: {issue['issue']}")
    
    print("\n✅ Validation and standardization complete!")
    
    # Save detailed report to file
    report_path = os.path.join(root_directory, "csv_validation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CSV Validation and Standardization Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total files processed: {stats['files_processed']}\n")
        f.write(f"Files with issues: {stats['files_with_issues']}\n")
        f.write(f"Files fixed: {stats['files_fixed']}\n")
        f.write(f"Files sorted: {stats['files_sorted']}\n")
        f.write(f"Test Names fixed: {stats['test_names_fixed']}\n")
        f.write(f"Subject Names fixed: {stats['subject_names_fixed']}\n")
        f.write(f"Empty files found: {len(stats['empty_files'])}\n\n")
        
        if stats['empty_files']:
            f.write("Empty Files:\n")
            for file in stats['empty_files']:
                f.write(f"  - {file}\n")
            f.write("\n")
        
        if stats['fixes_applied']:
            f.write("Fixes Applied:\n")
            for fix in stats['fixes_applied']:
                f.write(f"\n{fix['file']}:\n")
                f.write(f"  {fix['fix']}\n")
            f.write("\n")
        
        f.write("All Unique Field Names:\n")
        for field in sorted(all_field_names):
            f.write(f"  - {field}\n")
        
        f.write("\n\nFile Details:\n")
        for file, metadata in file_metadata.items():
            f.write(f"\n{file}:\n")
            f.write(f"  Subject Names: {metadata['subject_names']}\n")
            f.write(f"  Test Names: {metadata['test_names']}\n")
            f.write(f"  Row Count: {metadata['row_count']}\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    return stats

if __name__ == "__main__":
    # Get directory from command line argument or use current directory
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."
    
    # Check if --fix flag is provided
    fix_issues = "--fix" in sys.argv
    
    # Change to the specified directory
    if os.path.exists(directory):
        print(f"Validating CSV files in: {os.path.abspath(directory)}")
        if fix_issues:
            print("🔧 Fix mode enabled - will apply corrections to files")
        else:
            print("👀 Read-only mode - will only report issues without fixing")
        
        validate_and_standardize_csvs(directory, fix_issues=fix_issues)
    else:
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1) 