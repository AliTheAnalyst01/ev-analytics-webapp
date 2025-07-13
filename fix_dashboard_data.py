#!/usr/bin/env python3
"""
Fix Dashboard Data Issues
========================

Ensures data is properly formatted for the enhanced dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path

def fix_data_file():
    """Fix data formatting issues for dashboard"""
    print("🔧 Fixing data file for enhanced dashboard...")
    
    project_root = Path.cwd()
    
    # Try to load the data file
    source_files = [
        project_root / "data" / "raw" / "electric_vehicles_spec_2025.csv.csv",
        project_root / "data" / "raw" / "electric_vehicles_spec_2025.csv"
    ]
    
    df = None
    source_file = None
    
    for file_path in source_files:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                source_file = file_path
                print(f"✅ Loaded data from: {file_path}")
                print(f"   Shape: {df.shape}")
                break
            except Exception as e:
                print(f"❌ Could not load {file_path}: {e}")
                continue
    
    if df is None:
        print("❌ No valid data file found!")
        return False
    
    # Data validation and cleaning
    print("\n🔍 Validating and cleaning data...")
    
    # Check for essential columns
    required_columns = ['brand', 'model']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    
    # Convert numeric columns
    numeric_columns = [
        'top_speed_kmh', 'battery_capacity_kWh', 'number_of_cells', 'torque_nm',
        'efficiency_wh_per_km', 'range_km', 'acceleration_0_100_s',
        'fast_charging_power_kw_dc', 'towing_capacity_kg', 'cargo_volume_l',
        'seats', 'length_mm', 'width_mm', 'height_mm'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric, replacing non-numeric values with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"   ✅ Converted {col} to numeric")
    
    # Fill missing values for essential numeric columns
    essential_numeric = ['battery_capacity_kWh', 'range_km', 'efficiency_wh_per_km']
    
    for col in essential_numeric:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"   🔧 Filled {missing_count} missing values in {col} with median: {median_val}")
    
    # Ensure string columns are properly formatted
    string_columns = ['brand', 'model', 'battery_type', 'fast_charge_port', 
                     'drivetrain', 'segment', 'car_body_type', 'source_url']
    
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
            # Replace 'nan' strings with actual NaN
            df[col] = df[col].replace('nan', np.nan)
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Save the cleaned data to both possible locations
    target_files = [
        project_root / "data" / "raw" / "electric_vehicles_spec_2025.csv",
        project_root / "data" / "processed" / "ev_data_clean.csv"
    ]
    
    # Ensure directories exist
    for target_file in target_files:
        target_file.parent.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for target_file in target_files:
        try:
            df.to_csv(target_file, index=False)
            print(f"✅ Saved cleaned data to: {target_file}")
            success_count += 1
        except Exception as e:
            print(f"❌ Could not save to {target_file}: {e}")
    
    if success_count > 0:
        print(f"\n🎉 Data cleaning completed successfully!")
        print(f"   Final shape: {df.shape}")
        print(f"   Numeric columns: {len([col for col in numeric_columns if col in df.columns])}")
        print(f"   Files saved: {success_count}")
        
        # Show data summary
        print("\n📊 Data Summary:")
        print(f"   Total vehicles: {len(df)}")
        print(f"   Unique brands: {df['brand'].nunique()}")
        if 'range_km' in df.columns:
            print(f"   Range: {df['range_km'].min():.0f} - {df['range_km'].max():.0f} km")
        if 'battery_capacity_kWh' in df.columns:
            print(f"   Battery: {df['battery_capacity_kWh'].min():.1f} - {df['battery_capacity_kWh'].max():.1f} kWh")
        
        return True
    else:
        print("❌ Could not save cleaned data!")
        return False

if __name__ == "__main__":
    success = fix_data_file()
    
    if success:
        print("\n🚀 Ready to launch enhanced dashboard!")
        print("Run: streamlit run dashboard/enhanced_app.py")
    else:
        print("\n❌ Data fixing failed. Please check the data files manually.")