#!/usr/bin/env python3
"""
One-time script to standardize participant numbers across all facemesh CSV files.

This script:
1. Finds all CSV files containing facemesh data
2. Standardizes participant number formats (e1, E1, 1 -> e1)
3. Updates files in-place with backup
4. Reports all changes made

Usage:
    python standardize_participant_numbers.py

Author: AI Assistant for Facemesh Project
Date: 2025-01-28
"""

import pandas as pd
import numpy as np
import os
import glob
import re
import shutil
from datetime import datetime
from pathlib import Path

# Configuration
BACKUP_SUFFIX = ".backup"
DRY_RUN = False  # Set to True to see changes without making them

def standardize_participant_number(participant_str):
    """
    Standardize participant numbers to format 'eN' where N is the number.
    
    Input formats:
    - 'e1' -> 'e1' (already correct)
    - 'E1' -> 'e1' (convert to lowercase)
    - '1'  -> 'e1' (add 'e' prefix)
    - 'e-13' -> 'e13' (remove dashes)
    
    Args:
        participant_str: Input participant identifier
        
    Returns:
        str: Standardized participant identifier
    """
    if pd.isna(participant_str):
        return participant_str
    
    # Convert to string and strip whitespace
    participant = str(participant_str).strip()
    
    # Skip if empty
    if not participant:
        return participant
    
    # Remove any dashes
    participant = participant.replace('-', '')
    
    # Extract number using regex
    number_match = re.search(r'(\d+)', participant)
    if number_match:
        number = number_match.group(1)
        return f'e{number}'
    else:
        print(f"    Warning: Could not extract number from '{participant_str}' - keeping as is")
        return participant

def find_facemesh_files(root_dir="."):
    """Find all CSV files that contain facemesh data."""
    facemesh_files = []
    
    print(f"Searching for facemesh CSV files in: {os.path.abspath(root_dir)}")
    
    for csv_file in glob.glob(os.path.join(root_dir, "**/*.csv"), recursive=True):
        try:
            # Quick check for facemesh data by examining columns
            df_sample = pd.read_csv(csv_file, nrows=1)
            
            # Look for feat_N_x pattern (indicating facemesh data)
            feat_cols = [col for col in df_sample.columns if re.match(r'feat_\d+_[xyz]', col)]
            
            if len(feat_cols) > 0:
                facemesh_files.append(csv_file)
                print(f"  Found: {csv_file} ({len(feat_cols)} feature columns)")
                
        except Exception as e:
            # Skip files that can't be read
            continue
    
    return facemesh_files

def find_participant_columns(df):
    """Find columns that might contain participant identifiers."""
    participant_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['participant', 'subject']):
            participant_cols.append(col)
    
    return participant_cols

def process_file(file_path):
    """
    Process a single CSV file to standardize participant numbers.
    
    Returns:
        dict: Processing results
    """
    print(f"\nProcessing: {file_path}")
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
        original_df = df.copy()
        
        print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Find participant columns
        participant_cols = find_participant_columns(df)
        
        if not participant_cols:
            print(f"  No participant columns found - skipping")
            return {
                'file': file_path,
                'status': 'skipped',
                'reason': 'no_participant_columns',
                'changes': 0
            }
        
        print(f"  Found participant columns: {participant_cols}")
        
        # Track changes
        total_changes = 0
        changes_by_column = {}
        
        # Process each participant column
        for col in participant_cols:
            print(f"  Processing column: {col}")
            
            # Get unique values before standardization
            original_values = df[col].dropna().unique()
            print(f"    Original values: {sorted(original_values)}")
            
            # Apply standardization
            standardized = df[col].apply(standardize_participant_number)
            
            # Count changes
            changes = sum(original_df[col] != standardized)
            total_changes += changes
            changes_by_column[col] = changes
            
            # Update the dataframe
            df[col] = standardized
            
            # Get unique values after standardization
            new_values = df[col].dropna().unique()
            print(f"    Standardized: {sorted(new_values)}")
            print(f"    Changes made: {changes}")
        
        # Save results if changes were made and not in dry run mode
        if total_changes > 0 and not DRY_RUN:
            # Create backup
            backup_path = file_path + BACKUP_SUFFIX
            shutil.copy2(file_path, backup_path)
            print(f"  Backup created: {backup_path}")
            
            # Save updated file
            df.to_csv(file_path, index=False)
            print(f"  File updated with {total_changes} changes")
        elif total_changes > 0 and DRY_RUN:
            print(f"  DRY RUN: Would make {total_changes} changes")
        else:
            print(f"  No changes needed")
        
        return {
            'file': file_path,
            'status': 'success',
            'changes': total_changes,
            'changes_by_column': changes_by_column,
            'participant_cols': participant_cols
        }
        
    except Exception as e:
        print(f"  Error: {e}")
        return {
            'file': file_path,
            'status': 'error',
            'error': str(e),
            'changes': 0
        }

def main():
    """Main execution function."""
    print("=" * 60)
    print("    PARTICIPANT NUMBER STANDARDIZATION SCRIPT")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if DRY_RUN:
        print("🔍 DRY RUN MODE - No files will be modified")
    else:
        print("⚠️  LIVE MODE - Files will be modified (backups created)")
    
    # Find all facemesh files
    facemesh_files = find_facemesh_files()
    
    if not facemesh_files:
        print("\n❌ No facemesh CSV files found!")
        return
    
    print(f"\n📁 Found {len(facemesh_files)} facemesh files to process")
    
    # Process each file
    results = []
    for file_path in facemesh_files:
        result = process_file(file_path)
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("                    SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] == 'error']
    skipped = [r for r in results if r['status'] == 'skipped']
    
    total_changes = sum(r.get('changes', 0) for r in results)
    
    print(f"📊 Processing Results:")
    print(f"  • Total files processed: {len(results)}")
    print(f"  • Successful: {len(successful)}")
    print(f"  • Errors: {len(errors)}")
    print(f"  • Skipped: {len(skipped)}")
    print(f"  • Total changes made: {total_changes}")
    
    if successful:
        print(f"\n✅ Successfully processed files:")
        for result in successful:
            if result['changes'] > 0:
                print(f"  • {os.path.basename(result['file'])}: {result['changes']} changes")
            else:
                print(f"  • {os.path.basename(result['file'])}: no changes needed")
    
    if errors:
        print(f"\n❌ Errors encountered:")
        for result in errors:
            print(f"  • {os.path.basename(result['file'])}: {result.get('error', 'Unknown error')}")
    
    if skipped:
        print(f"\n⏭️  Skipped files:")
        for result in skipped:
            print(f"  • {os.path.basename(result['file'])}: {result.get('reason', 'Unknown reason')}")
    
    # Backup information
    if total_changes > 0 and not DRY_RUN:
        print(f"\n💾 Backup files created (suffix: {BACKUP_SUFFIX})")
        print(f"   To restore original files: rename .backup files back to .csv")
    
    print(f"\n🎉 Participant number standardization complete!")
    print(f"   All participant IDs should now be in 'eN' format (e.g., e1, e2, e13)")
    
    if DRY_RUN:
        print(f"\n🔍 This was a DRY RUN - set DRY_RUN=False to apply changes")

if __name__ == "__main__":
    # Safety check - ask for confirmation if not in dry run mode
    if not DRY_RUN:
        print("⚠️  WARNING: This script will modify CSV files in-place!")
        print("   Backups will be created, but please ensure you have your own backups.")
        response = input("\nDo you want to continue? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            print("Operation cancelled.")
            exit()
    
    main() 