#!/usr/bin/env python3
"""
Fix data file by copying content properly
"""

import pandas as pd
from pathlib import Path

def fix_data_file():
    """Copy data from .csv.csv to .csv file"""
    project_root = Path.cwd()
    
    source_file = project_root / "data" / "raw" / "electric_vehicles_spec_2025.csv.csv"
    target_file = project_root / "data" / "raw" / "electric_vehicles_spec_2025.csv"
    
    print(f"Copying data from: {source_file}")
    print(f"To: {target_file}")
    
    try:
        # Read the source data
        df = pd.read_csv(source_file)
        print(f"✅ Loaded data: {len(df)} rows, {len(df.columns)} columns")
        
        # Save to target file
        df.to_csv(target_file, index=False)
        print(f"✅ Saved data to: {target_file}")
        
        # Verify the copy
        df_verify = pd.read_csv(target_file)
        print(f"✅ Verified: {len(df_verify)} rows loaded from new file")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = fix_data_copy()
    if success:
        print("\n🎉 Data file fixed successfully!")
        print("You can now refresh the dashboard.")
    else:
        print("\n❌ Failed to fix data file.")