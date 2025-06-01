import os
import pandas as pd
import glob

def fix_session_labels():
    """
    Fix session labels in all CSV files in the current directory.
    Changes 'e26 session' to 'session' (and similar for all sessions).
    """
    # Get all CSV files in the current directory
    csv_files = glob.glob('*.csv')
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process.")
    
    for csv_file in csv_files:
        try:
            print(f"\nProcessing: {csv_file}")
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Check if 'Test Name' column exists
            if 'Test Name' in df.columns:
                # Count how many rows need to be fixed
                needs_fix = df['Test Name'].str.contains('e26 session', na=False).sum()
                
                if needs_fix > 0:
                    # Replace 'e26 session' with just 'session' in the Test Name column
                    df['Test Name'] = df['Test Name'].str.replace('e26 session', 'session', regex=False)
                    
                    # Save the modified DataFrame back to the same file
                    df.to_csv(csv_file, index=False)
                    
                    print(f"  ✓ Fixed {needs_fix} rows in {csv_file}")
                else:
                    print(f"  - No changes needed for {csv_file}")
            else:
                print(f"  ! Warning: 'Test Name' column not found in {csv_file}")
                
        except Exception as e:
            print(f"  ✗ Error processing {csv_file}: {str(e)}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    # Change to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Run the fix
    fix_session_labels() 