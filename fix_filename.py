#!/usr/bin/env python3
"""
Quick fix script to rename the data file and verify structure
"""

import os
import shutil
from pathlib import Path
import pandas as pd

def fix_filename():
    project_root = Path.cwd()
    data_raw_path = project_root / "data" / "raw"
    
    # Source and target files
    source_file = data_raw_path / "electric_vehicles_spec_2025.csv.csv"
    target_file = data_raw_path / "electric_vehicles_spec_2025.csv"
    
    print("🔧 Fixing filename issue...")
    print(f"Source: {source_file}")
    print(f"Target: {target_file}")
    
    if source_file.exists():
        # Remove target if it exists
        if target_file.exists():
            target_file.unlink()
            print("✅ Removed existing target file")
        
        # Rename the file
        source_file.rename(target_file)
        print("✅ File renamed successfully!")
        
        # Verify the file content
        try:
            df = pd.read_csv(target_file)
            print(f"✅ Data verification successful:")
            print(f"   - Shape: {df.shape}")
            print(f"   - Columns: {len(df.columns)}")
            print(f"   - Sample columns: {list(df.columns[:5])}")
            
            return True
        except Exception as e:
            print(f"❌ Data verification failed: {e}")
            return False
    else:
        print("❌ Source file not found!")
        return False

if __name__ == "__main__":
    success = fix_filename()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")