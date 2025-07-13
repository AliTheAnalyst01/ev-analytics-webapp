#!/usr/bin/env python3
"""
Copy the data file content to correctly named file
"""

import shutil
from pathlib import Path

def copy_data_file():
    project_root = Path.cwd()
    data_raw_path = project_root / "data" / "raw"
    
    # Source and target files
    source_file = data_raw_path / "electric_vehicles_spec_2025.csv.csv"
    target_file = data_raw_path / "electric_vehicles_spec_2025.csv"
    
    print("📁 Copying data file...")
    print(f"Source: {source_file}")
    print(f"Target: {target_file}")
    
    try:
        # Copy the file content
        shutil.copy2(source_file, target_file)
        print("✅ File copied successfully!")
        
        # Verify file sizes match
        source_size = source_file.stat().st_size
        target_size = target_file.stat().st_size
        
        print(f"Source size: {source_size} bytes")
        print(f"Target size: {target_size} bytes")
        
        if source_size == target_size:
            print("✅ File sizes match - copy successful!")
            return True
        else:
            print("❌ File sizes don't match!")
            return False
            
    except Exception as e:
        print(f"❌ Copy failed: {e}")
        return False

if __name__ == "__main__":
    copy_data_file()