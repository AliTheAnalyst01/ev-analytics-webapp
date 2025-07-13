#!/usr/bin/env python3
"""
Create sample processed data for the dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_processed_data():
    """Create sample processed data with features"""
    
    # Read the raw data
    raw_path = Path("data/raw/electric_vehicles_spec_2025.csv.csv")
    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        raw_path = Path("data/raw/electric_vehicles_spec_2025.csv")
        df = pd.read_csv(raw_path)
    
    print(f"Loaded raw data: {df.shape}")
    
    # Create some engineered features
    if 'battery_capacity_kWh' in df.columns and 'range_km' in df.columns:
        df['range_per_kwh'] = df['range_km'] / df['battery_capacity_kWh']
        print("Created: range_per_kwh")
    
    if 'efficiency_wh_per_km' in df.columns:
        df['efficiency_kwh_per_100km'] = df['efficiency_wh_per_km'] / 10
        print("Created: efficiency_kwh_per_100km")
    
    if 'top_speed_kmh' in df.columns and 'acceleration_0_100_s' in df.columns:
        df['performance_ratio'] = df['top_speed_kmh'] / df['acceleration_0_100_s']
        print("Created: performance_ratio")
    
    if 'torque_nm' in df.columns and 'battery_capacity_kWh' in df.columns:
        df['torque_per_kwh'] = df['torque_nm'] / df['battery_capacity_kWh']
        print("Created: torque_per_kwh")
    
    # Create categorical features
    if 'range_km' in df.columns:
        df['range_category'] = pd.cut(df['range_km'], 
                                     bins=[0, 300, 500, 700, float('inf')],
                                     labels=['Short', 'Medium', 'Long', 'Very Long'])
        print("Created: range_category")
    
    # Ensure processed directory exists
    processed_path = Path("data/processed")
    processed_path.mkdir(exist_ok=True)
    
    # Save processed data
    output_file = processed_path / "ev_data_engineered.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Saved processed data: {output_file}")
    print(f"Final shape: {df.shape}")
    print(f"New columns: {df.shape[1] - 22}")  # Original had 22 columns
    
    return True

if __name__ == "__main__":
    create_processed_data()