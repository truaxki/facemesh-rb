import pandas as pd
import sys

if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = 'read/e1/e1-baseline-rb5-rel.csv'

print(f"Checking file: {file_path}")
df = pd.read_csv(file_path)
print(f"Total columns: {len(df.columns)}")
print(f"Total rows: {len(df)}")

# Check for different feature types
rb5_diff_cols = [c for c in df.columns if 'rb5_diff' in c and 'rel' not in c]
rb5_rel_mag_cols = [c for c in df.columns if 'rb5_rel_mag' in c]
rb5_rel_diff_cols = [c for c in df.columns if 'rb5_rel_diff' in c]

print(f"\nFeature counts:")
print(f"rb5_diff columns: {len(rb5_diff_cols)}")
print(f"rb5_rel_mag columns: {len(rb5_rel_mag_cols)}")
print(f"rb5_rel_diff columns: {len(rb5_rel_diff_cols)}")

# Show samples
print(f"\nSample rb5_diff columns: {rb5_diff_cols[:3]}")
print(f"Sample rb5_rel_mag columns: {rb5_rel_mag_cols[:3]}")
print(f"Sample rb5_rel_diff columns: {rb5_rel_diff_cols[:3]}")

# Check for nose and cheek features
nose_cheek_landmarks = [1, 2, 98, 327, 205, 425]
print(f"\nChecking nose+cheek landmarks {nose_cheek_landmarks}:")
for landmark in nose_cheek_landmarks:
    mag_col = f'feat_{landmark}_rb5_rel_mag'
    if mag_col in df.columns:
        print(f"  ✓ {mag_col} found")
    else:
        print(f"  ✗ {mag_col} NOT found") 