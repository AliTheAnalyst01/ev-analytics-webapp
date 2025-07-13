#!/usr/bin/env python3
"""
Advanced EV Prediction Models
============================

10 comprehensive prediction models beyond just range prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedEVPredictions:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.predictions = {}
        
    def load_and_prepare_data(self):
        """Load and prepare EV data with enhanced features"""
        # Load data
        df = pd.read_csv('data/raw/electric_vehicles_spec_2025.csv.csv')
        
        # Create enhanced features for predictions
        self.create_enhanced_features(df)
        
        return df
    
    def create_enhanced_features(self, df):
        """Create advanced features for prediction models"""
        
        # 1. Efficiency metrics
        if 'battery_capacity_kWh' in df.columns and 'range_km' in df.columns:
            df['range_per_kwh'] = df['range_km'] / df['battery_capacity_kWh']
            df['energy_density'] = df['battery_capacity_kWh'] / (df['length_mm'] * df['width_mm'] * df['height_mm'] / 1e9)
        
        # 2. Performance metrics
        if 'top_speed_kmh' in df.columns and 'acceleration_0_100_s' in df.columns:
            df['performance_index'] = df['top_speed_kmh'] / df['acceleration_0_100_s']
            df['power_estimate'] = df['torque_nm'] * df['top_speed_kmh'] / 100  # Rough power estimate
        
        # 3. Charging metrics
        if 'fast_charging_power_kw_dc' in df.columns and 'battery_capacity_kWh' in df.columns:
            df['charging_c_rate'] = df['fast_charging_power_kw_dc'] / df['battery_capacity_kWh']
            df['charge_time_estimate'] = df['battery_capacity_kWh'] * 0.7 / df['fast_charging_power_kw_dc']  # 10-80% charge time
        
        # 4. Size and practicality
        if all(col in df.columns for col in ['length_mm', 'width_mm', 'height_mm']):
            df['vehicle_volume'] = df['length_mm'] * df['width_mm'] * df['height_mm'] / 1e9  # m³
            df['cargo_ratio'] = df['cargo_volume_l'] / df['vehicle_volume'] / 1000  # Cargo efficiency
        
        # 5. Brand premium indicators
        luxury_brands = ['Tesla', 'Mercedes', 'BMW', 'Audi', 'Porsche', 'Jaguar', 'Volvo', 'Genesis']
        df['is_luxury_brand'] = df['brand'].isin(luxury_brands).astype(int)
        
        # 6. Market positioning
        df['is_performance_vehicle'] = ((df['acceleration_0_100_s'] < 5) | (df['top_speed_kmh'] > 200)).astype(int)
        df['is_efficiency_focused'] = (df['efficiency_wh_per_km'] < 150).astype(int)
        
        return df
    
    def model_1_efficiency_prediction(self, df):
        """Model 1: Vehicle Efficiency Prediction (Wh/km)"""
        print("🔋 Training Model 1: Vehicle Efficiency Prediction")
        
        features = ['battery_capacity_kWh', 'range_km', 'top_speed_kmh', 'acceleration_0_100_s', 
                   'torque_nm', 'length_mm', 'width_mm', 'height_mm', 'is_luxury_brand']
        
        # Clean data
        model_df = df[features + ['efficiency_wh_per_km']].dropna()
        
        X = model_df[features]
        y = model_df['efficiency_wh_per_km']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        self.models['efficiency_prediction'] = model
        self.predictions['efficiency_prediction'] = {
            'r2_score': r2,
            'mae': mae,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'description': 'Predicts vehicle efficiency (Wh/km) based on specifications'
        }
        
        print(f"   R² Score: {r2:.3f}, MAE: {mae:.1f} Wh/km")
        return model
    
    def model_2_price_estimation(self, df):
        """Model 2: Purchase Price Estimation"""
        print("💰 Training Model 2: Purchase Price Estimation")
        
        # Create synthetic price data based on realistic patterns
        df['estimated_price_eur'] = self.create_synthetic_prices(df)
        
        features = ['battery_capacity_kWh', 'range_km', 'top_speed_kmh', 'acceleration_0_100_s',
                   'torque_nm', 'fast_charging_power_kw_dc', 'is_luxury_brand', 'performance_index']
        
        model_df = df[features + ['estimated_price_eur']].dropna()
        
        X = model_df[features]
        y = model_df['estimated_price_eur']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        self.models['price_estimation'] = model
        self.predictions['price_estimation'] = {
            'r2_score': r2,
            'mae': mae,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'description': 'Estimates vehicle purchase price based on specifications and brand'
        }
        
        print(f"   R² Score: {r2:.3f}, MAE: €{mae:.0f}")
        return model
    
    def model_3_charging_time_prediction(self, df):
        """Model 3: Charging Time Prediction"""
        print("⚡ Training Model 3: Charging Time Prediction")
        
        features = ['battery_capacity_kWh', 'fast_charging_power_kw_dc', 'charging_c_rate']
        
        model_df = df[features + ['charge_time_estimate']].dropna()
        
        X = model_df[features]
        y = model_df['charge_time_estimate']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        self.models['charging_time'] = model
        self.predictions['charging_time'] = {
            'r2_score': r2,
            'mae': mae,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'description': 'Predicts 10-80% charging time in hours'
        }
        
        print(f"   R² Score: {r2:.3f}, MAE: {mae:.2f} hours")
        return model
    
    def model_4_range_gap_prediction(self, df):
        """Model 4: Real-World vs EPA Range Gap"""
        print("📊 Training Model 4: Real-World vs EPA Range Gap")
        
        # Create synthetic range gap data (real-world typically 15-25% lower)
        df['real_world_range'] = df['range_km'] * (0.75 + 0.15 * np.random.random(len(df)))
        df['range_gap_percent'] = ((df['range_km'] - df['real_world_range']) / df['range_km']) * 100
        
        features = ['efficiency_wh_per_km', 'top_speed_kmh', 'battery_capacity_kWh', 
                   'is_performance_vehicle', 'vehicle_volume']
        
        model_df = df[features + ['range_gap_percent']].dropna()
        
        X = model_df[features]
        y = model_df['range_gap_percent']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        self.models['range_gap'] = model
        self.predictions['range_gap'] = {
            'r2_score': r2,
            'mae': mae,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'description': 'Predicts percentage gap between EPA and real-world range'
        }
        
        print(f"   R² Score: {r2:.3f}, MAE: {mae:.1f}%")
        return model
    
    def model_5_market_segment_classification(self, df):
        """Model 5: Advanced Market Segment Classification"""
        print("🎯 Training Model 5: Market Segment Classification")
        
        # Enhanced segment classification based on price, performance, and luxury
        df['market_segment'] = self.create_market_segments(df)
        
        features = ['battery_capacity_kWh', 'range_km', 'top_speed_kmh', 'acceleration_0_100_s',
                   'estimated_price_eur', 'is_luxury_brand', 'performance_index']
        
        model_df = df[features + ['market_segment']].dropna()
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(model_df['market_segment'])
        
        X = model_df[features]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['market_segment'] = model
        self.encoders['market_segment'] = le
        self.predictions['market_segment'] = {
            'accuracy': accuracy,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'classes': le.classes_.tolist(),
            'description': 'Classifies vehicles into market segments: Budget, Mainstream, Premium, Luxury, Performance'
        }
        
        print(f"   Accuracy: {accuracy:.3f}")
        return model
    
    def model_6_maintenance_cost_prediction(self, df):
        """Model 6: Maintenance Cost Prediction"""
        print("🔧 Training Model 6: Maintenance Cost Prediction")
        
        # Create synthetic maintenance costs
        df['annual_maintenance_eur'] = self.create_maintenance_costs(df)
        
        features = ['battery_capacity_kWh', 'is_luxury_brand', 'top_speed_kmh', 
                   'estimated_price_eur', 'vehicle_volume']
        
        model_df = df[features + ['annual_maintenance_eur']].dropna()
        
        X = model_df[features]
        y = model_df['annual_maintenance_eur']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        self.models['maintenance_cost'] = model
        self.predictions['maintenance_cost'] = {
            'r2_score': r2,
            'mae': mae,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'description': 'Predicts annual maintenance cost in EUR'
        }
        
        print(f"   R² Score: {r2:.3f}, MAE: €{mae:.0f}")
        return model
    
    def model_7_resale_value_prediction(self, df):
        """Model 7: Resale Value Estimation"""
        print("📈 Training Model 7: Resale Value Estimation")
        
        # Create 3-year resale values (typically 50-70% of original price)
        df['resale_value_3yr_eur'] = df['estimated_price_eur'] * (0.5 + 0.2 * np.random.random(len(df)))
        
        features = ['estimated_price_eur', 'range_km', 'efficiency_wh_per_km', 
                   'is_luxury_brand', 'battery_capacity_kWh']
        
        model_df = df[features + ['resale_value_3yr_eur']].dropna()
        
        X = model_df[features]
        y = model_df['resale_value_3yr_eur']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        self.models['resale_value'] = model
        self.predictions['resale_value'] = {
            'r2_score': r2,
            'mae': mae,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'description': 'Predicts 3-year resale value in EUR'
        }
        
        print(f"   R² Score: {r2:.3f}, MAE: €{mae:.0f}")
        return model
    
    def model_8_environmental_impact_score(self, df):
        """Model 8: Environmental Impact Score"""
        print("🌱 Training Model 8: Environmental Impact Score")
        
        # Create environmental impact score (0-100, higher is better)
        df['env_impact_score'] = self.create_environmental_scores(df)
        
        features = ['efficiency_wh_per_km', 'battery_capacity_kWh', 'vehicle_volume', 'range_km']
        
        model_df = df[features + ['env_impact_score']].dropna()
        
        X = model_df[features]
        y = model_df['env_impact_score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        self.models['env_impact'] = model
        self.predictions['env_impact'] = {
            'r2_score': r2,
            'mae': mae,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'description': 'Environmental impact score (0-100, higher is better)'
        }
        
        print(f"   R² Score: {r2:.3f}, MAE: {mae:.1f} points")
        return model
    
    def model_9_total_cost_ownership(self, df):
        """Model 9: Total Cost of Ownership"""
        print("💼 Training Model 9: Total Cost of Ownership")
        
        # Calculate 5-year TCO
        df['tco_5yr_eur'] = (df['estimated_price_eur'] + 
                             df['annual_maintenance_eur'] * 5 + 
                             (df['efficiency_wh_per_km'] * 15000 * 5 * 0.3 / 1000))  # Energy costs
        
        features = ['estimated_price_eur', 'efficiency_wh_per_km', 'annual_maintenance_eur',
                   'range_km', 'is_luxury_brand']
        
        model_df = df[features + ['tco_5yr_eur']].dropna()
        
        X = model_df[features]
        y = model_df['tco_5yr_eur']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        self.models['tco'] = model
        self.predictions['tco'] = {
            'r2_score': r2,
            'mae': mae,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'description': 'Predicts 5-year total cost of ownership in EUR'
        }
        
        print(f"   R² Score: {r2:.3f}, MAE: €{mae:.0f}")
        return model
    
    def model_10_reliability_score(self, df):
        """Model 10: Vehicle Reliability Score"""
        print("🛡️ Training Model 10: Vehicle Reliability Score")
        
        # Create reliability scores based on brand, complexity, technology maturity
        df['reliability_score'] = self.create_reliability_scores(df)
        
        features = ['battery_capacity_kWh', 'is_luxury_brand', 'number_of_cells',
                   'fast_charging_power_kw_dc', 'top_speed_kmh']
        
        model_df = df[features + ['reliability_score']].dropna()
        
        X = model_df[features]
        y = model_df['reliability_score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        self.models['reliability'] = model
        self.predictions['reliability'] = {
            'r2_score': r2,
            'mae': mae,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'description': 'Reliability score (0-100, higher is better)'
        }
        
        print(f"   R² Score: {r2:.3f}, MAE: {mae:.1f} points")
        return model
    
    def create_synthetic_prices(self, df):
        """Create realistic price estimates"""
        base_price = 25000  # Base price in EUR
        
        prices = (base_price + 
                 df['battery_capacity_kWh'] * 400 +  # €400 per kWh
                 df['range_km'] * 20 +  # Range premium
                 df['is_luxury_brand'] * 15000 +  # Luxury premium
                 df['performance_index'] * 100 +  # Performance premium
                 np.random.normal(0, 5000, len(df)))  # Random variation
        
        return np.clip(prices, 15000, 150000)  # Reasonable bounds
    
    def create_market_segments(self, df):
        """Create market segment classifications"""
        segments = []
        for _, row in df.iterrows():
            if row['estimated_price_eur'] < 30000:
                segments.append('Budget')
            elif row['estimated_price_eur'] < 50000:
                if row['is_luxury_brand']:
                    segments.append('Premium')
                else:
                    segments.append('Mainstream')
            elif row['estimated_price_eur'] < 80000:
                if row['is_performance_vehicle']:
                    segments.append('Performance')
                else:
                    segments.append('Premium')
            else:
                segments.append('Luxury')
        
        return segments
    
    def create_maintenance_costs(self, df):
        """Create maintenance cost estimates"""
        base_cost = 300  # Base annual maintenance
        
        costs = (base_cost +
                df['battery_capacity_kWh'] * 2 +  # Larger battery = more complexity
                df['is_luxury_brand'] * 400 +  # Luxury premium
                df['top_speed_kmh'] * 1.5 +  # Performance cars cost more
                np.random.normal(0, 100, len(df)))  # Random variation
        
        return np.clip(costs, 200, 2000)
    
    def create_environmental_scores(self, df):
        """Create environmental impact scores"""
        # Higher efficiency and smaller size = better score
        scores = (100 - (df['efficiency_wh_per_km'] - 100) / 3 -  # Efficiency factor
                 (df['vehicle_volume'] - 10) * 2 +  # Size penalty
                 np.random.normal(0, 5, len(df)))  # Random variation
        
        return np.clip(scores, 0, 100)
    
    def create_reliability_scores(self, df):
        """Create reliability scores"""
        # Based on brand reputation, complexity, technology maturity
        reliable_brands = ['Toyota', 'Hyundai', 'Kia', 'Nissan', 'BMW', 'Mercedes']
        
        scores = []
        for _, row in df.iterrows():
            base_score = 85 if row['brand'] in reliable_brands else 75
            
            # Adjust for complexity
            complexity_penalty = (row.get('number_of_cells', 100) - 100) / 10
            speed_penalty = max(0, (row['top_speed_kmh'] - 180) / 10)
            
            score = base_score - complexity_penalty - speed_penalty + np.random.normal(0, 5)
            scores.append(score)
        
        return np.clip(scores, 50, 100)
    
    def train_all_models(self):
        """Train all 10 prediction models"""
        print("🚗⚡ TRAINING 10 ADVANCED EV PREDICTION MODELS")
        print("=" * 60)
        
        df = self.load_and_prepare_data()
        
        # Train all models
        self.model_1_efficiency_prediction(df)
        self.model_2_price_estimation(df)
        self.model_3_charging_time_prediction(df)
        self.model_4_range_gap_prediction(df)
        self.model_5_market_segment_classification(df)
        self.model_6_maintenance_cost_prediction(df)
        self.model_7_resale_value_prediction(df)
        self.model_8_environmental_impact_score(df)
        self.model_9_total_cost_ownership(df)
        self.model_10_reliability_score(df)
        
        # Save all models and results
        self.save_models()
        
        print("\n🎉 ALL 10 MODELS TRAINED SUCCESSFULLY!")
        return self.predictions
    
    def save_models(self):
        """Save all models and predictions"""
        import joblib
        import json
        from pathlib import Path
        
        # Save models
        models_dir = Path("models/advanced")
        models_dir.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, models_dir / f"{name}_model.pkl")
        
        # Save encoders
        for name, encoder in self.encoders.items():
            joblib.dump(encoder, models_dir / f"{name}_encoder.pkl")
        
        # Save predictions summary
        with open(models_dir / "advanced_predictions.json", 'w') as f:
            json.dump(self.predictions, f, indent=2, default=str)
        
        print(f"💾 Saved {len(self.models)} models to {models_dir}")

if __name__ == "__main__":
    predictor = AdvancedEVPredictions()
    results = predictor.train_all_models()