#!/usr/bin/env python3
"""
Comprehensive EV Project Validation Script
=========================================

Validates data quality, model performance, and business logic
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

class EVProjectValidator:
    def __init__(self):
        self.project_root = Path.cwd()
        self.validation_results = {}
        self.data_quality_issues = []
        self.business_logic_issues = []
        self.technical_issues = []
        
    def load_data(self):
        """Load and return the EV dataset"""
        raw_path = self.project_root / "data" / "raw" / "electric_vehicles_spec_2025.csv.csv"
        if not raw_path.exists():
            raw_path = self.project_root / "data" / "raw" / "electric_vehicles_spec_2025.csv"
        
        self.df = pd.read_csv(raw_path)
        print(f"✅ Loaded dataset: {self.df.shape}")
        return self.df
    
    def validate_data_quality(self):
        """Comprehensive data quality validation"""
        print("\n🔍 VALIDATING DATA QUALITY")
        print("=" * 50)
        
        results = {}
        
        # Basic statistics
        results['total_records'] = len(self.df)
        results['total_features'] = len(self.df.columns)
        results['missing_values'] = self.df.isnull().sum().sum()
        results['duplicate_records'] = self.df.duplicated().sum()
        
        # Missing value analysis
        missing_by_column = self.df.isnull().sum().sort_values(ascending=False)
        results['missing_by_column'] = missing_by_column[missing_by_column > 0].to_dict()
        
        # Data type analysis
        results['data_types'] = self.df.dtypes.value_counts().to_dict()
        
        print(f"📊 Total Records: {results['total_records']}")
        print(f"📊 Total Features: {results['total_features']}")
        print(f"⚠️  Missing Values: {results['missing_values']}")
        print(f"🔄 Duplicate Records: {results['duplicate_records']}")
        
        if missing_by_column.sum() > 0:
            print("\n❌ Columns with missing values:")
            for col, count in missing_by_column[missing_by_column > 0].items():
                pct = (count / len(self.df)) * 100
                print(f"   {col}: {count} ({pct:.1f}%)")
                if pct > 20:
                    self.data_quality_issues.append(f"{col} has {pct:.1f}% missing values")
        
        self.validation_results['data_quality'] = results
        return results
    
    def validate_business_logic(self):
        """Validate business logic and physics constraints"""
        print("\n🧠 VALIDATING BUSINESS LOGIC")
        print("=" * 50)
        
        results = {}
        
        # Physics validation: Range vs Battery vs Efficiency
        if all(col in self.df.columns for col in ['range_km', 'battery_capacity_kWh', 'efficiency_wh_per_km']):
            # Calculate theoretical range based on battery and efficiency
            self.df['theoretical_range'] = (self.df['battery_capacity_kWh'] * 1000) / self.df['efficiency_wh_per_km']
            
            # Check for unrealistic discrepancies
            self.df['range_discrepancy'] = abs(self.df['range_km'] - self.df['theoretical_range'])
            self.df['range_discrepancy_pct'] = (self.df['range_discrepancy'] / self.df['range_km']) * 100
            
            large_discrepancies = self.df[self.df['range_discrepancy_pct'] > 20]
            results['physics_validation'] = {
                'records_with_large_discrepancy': len(large_discrepancies),
                'avg_discrepancy_pct': self.df['range_discrepancy_pct'].mean(),
                'max_discrepancy_pct': self.df['range_discrepancy_pct'].max()
            }
            
            print(f"⚡ Physics Check - Range vs Battery/Efficiency:")
            print(f"   Average discrepancy: {results['physics_validation']['avg_discrepancy_pct']:.1f}%")
            print(f"   Records with >20% discrepancy: {len(large_discrepancies)}")
            
            if len(large_discrepancies) > 10:
                self.business_logic_issues.append(f"{len(large_discrepancies)} vehicles have unrealistic range calculations")
        
        # Realistic value ranges
        numeric_columns = ['top_speed_kmh', 'battery_capacity_kWh', 'range_km', 'acceleration_0_100_s', 'efficiency_wh_per_km']
        expected_ranges = {
            'top_speed_kmh': (80, 300),
            'battery_capacity_kWh': (10, 200),
            'range_km': (50, 1000),
            'acceleration_0_100_s': (2, 20),
            'efficiency_wh_per_km': (100, 400)
        }
        
        outliers_summary = {}
        for col in numeric_columns:
            if col in self.df.columns:
                min_val, max_val = expected_ranges[col]
                outliers = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)]
                outliers_summary[col] = len(outliers)
                
                if len(outliers) > 0:
                    print(f"📈 {col}: {len(outliers)} outliers outside [{min_val}, {max_val}]")
                    if len(outliers) > 5:
                        self.business_logic_issues.append(f"{col} has {len(outliers)} outliers")
        
        results['outliers'] = outliers_summary
        
        # Brand/Model consistency
        brand_model_counts = self.df.groupby('brand').size().sort_values(ascending=False)
        results['brand_distribution'] = brand_model_counts.head(10).to_dict()
        
        print(f"\n🏭 Top 5 Brands by Model Count:")
        for brand, count in brand_model_counts.head(5).items():
            print(f"   {brand}: {count} models")
        
        self.validation_results['business_logic'] = results
        return results
    
    def validate_model_results(self):
        """Validate ML model performance metrics"""
        print("\n🤖 VALIDATING MODEL RESULTS")
        print("=" * 50)
        
        results = {}
        
        # Load model results
        model_results_path = self.project_root / "models" / "model_results.json"
        if model_results_path.exists():
            with open(model_results_path, 'r') as f:
                model_results = json.load(f)
            
            # Validate regression metrics
            if 'regression' in model_results:
                reg_results = model_results['regression']
                r2_scores = [model['val_metrics']['r2'] for model in reg_results.values() if 'val_metrics' in model]
                rmse_scores = [model['val_metrics']['rmse'] for model in reg_results.values() if 'val_metrics' in model]
                
                results['regression'] = {
                    'models_count': len(reg_results),
                    'best_r2': max(r2_scores) if r2_scores else 0,
                    'avg_r2': np.mean(r2_scores) if r2_scores else 0,
                    'best_rmse': min(rmse_scores) if rmse_scores else float('inf'),
                    'avg_rmse': np.mean(rmse_scores) if rmse_scores else 0
                }
                
                print(f"📈 Regression Models: {len(reg_results)} trained")
                print(f"   Best R² Score: {results['regression']['best_r2']:.3f}")
                print(f"   Best RMSE: {results['regression']['best_rmse']:.1f} km")
                
                # Validate reasonableness
                if results['regression']['best_r2'] < 0.7:
                    self.technical_issues.append("Best R² score below 0.7 suggests poor model performance")
                if results['regression']['best_rmse'] > 100:
                    self.technical_issues.append("RMSE above 100km suggests poor range prediction accuracy")
            
            # Validate classification metrics
            if 'classification' in model_results:
                cls_results = model_results['classification']
                f1_scores = [model['val_metrics']['f1'] for model in cls_results.values() if 'val_metrics' in model]
                acc_scores = [model['val_metrics']['accuracy'] for model in cls_results.values() if 'val_metrics' in model]
                
                results['classification'] = {
                    'models_count': len(cls_results),
                    'best_f1': max(f1_scores) if f1_scores else 0,
                    'avg_f1': np.mean(f1_scores) if f1_scores else 0,
                    'best_accuracy': max(acc_scores) if acc_scores else 0,
                    'avg_accuracy': np.mean(acc_scores) if acc_scores else 0
                }
                
                print(f"🎯 Classification Models: {len(cls_results)} trained")
                print(f"   Best F1 Score: {results['classification']['best_f1']:.3f}")
                print(f"   Best Accuracy: {results['classification']['best_accuracy']:.3f}")
                
                if results['classification']['best_f1'] < 0.6:
                    self.technical_issues.append("Best F1 score below 0.6 suggests poor classification performance")
            
            # Validate clustering metrics
            if 'clustering' in model_results:
                cluster_results = model_results['clustering']
                silhouette_scores = [model['metrics']['silhouette_score'] for model in cluster_results.values() if 'metrics' in model]
                
                results['clustering'] = {
                    'algorithms_count': len(cluster_results),
                    'best_silhouette': max(silhouette_scores) if silhouette_scores else 0,
                    'avg_silhouette': np.mean(silhouette_scores) if silhouette_scores else 0
                }
                
                print(f"🔗 Clustering Algorithms: {len(cluster_results)} tested")
                print(f"   Best Silhouette Score: {results['clustering']['best_silhouette']:.3f}")
                
                if results['clustering']['best_silhouette'] < 0.3:
                    self.technical_issues.append("Best silhouette score below 0.3 suggests poor clustering quality")
        
        else:
            print("❌ Model results file not found")
            self.technical_issues.append("Model results file missing")
        
        self.validation_results['model_performance'] = results
        return results
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n📋 GENERATING VALIDATION REPORT")
        print("=" * 50)
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'project_status': 'VALIDATED',
            'overall_score': 0,
            'validation_results': self.validation_results,
            'issues_found': {
                'data_quality': self.data_quality_issues,
                'business_logic': self.business_logic_issues,
                'technical': self.technical_issues
            },
            'recommendations': []
        }
        
        # Calculate overall score
        score = 100
        
        # Deduct points for issues
        score -= len(self.data_quality_issues) * 10
        score -= len(self.business_logic_issues) * 15
        score -= len(self.technical_issues) * 20
        
        # Bonus points for good performance
        if 'model_performance' in self.validation_results:
            if self.validation_results['model_performance'].get('regression', {}).get('best_r2', 0) > 0.85:
                score += 10
            if self.validation_results['model_performance'].get('classification', {}).get('best_f1', 0) > 0.85:
                score += 10
        
        report['overall_score'] = max(0, min(100, score))
        
        # Generate recommendations
        if len(self.data_quality_issues) > 0:
            report['recommendations'].append("Address data quality issues with missing values and outliers")
        
        if len(self.business_logic_issues) > 0:
            report['recommendations'].append("Review business logic constraints and physics validation")
        
        if len(self.technical_issues) > 0:
            report['recommendations'].append("Improve model performance through feature engineering and hyperparameter tuning")
        
        if report['overall_score'] >= 90:
            report['project_status'] = 'EXCELLENT'
        elif report['overall_score'] >= 70:
            report['project_status'] = 'GOOD'
        elif report['overall_score'] >= 50:
            report['project_status'] = 'NEEDS_IMPROVEMENT'
        else:
            report['project_status'] = 'REQUIRES_MAJOR_FIXES'
        
        # Save report
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"🎯 OVERALL PROJECT SCORE: {report['overall_score']}/100")
        print(f"📊 PROJECT STATUS: {report['project_status']}")
        print(f"📄 Full report saved: {report_path}")
        
        if len(self.data_quality_issues) + len(self.business_logic_issues) + len(self.technical_issues) == 0:
            print("🎉 NO CRITICAL ISSUES FOUND - PROJECT READY FOR ENHANCEMENT!")
        else:
            print(f"\n⚠️  ISSUES FOUND:")
            for issue in self.data_quality_issues + self.business_logic_issues + self.technical_issues:
                print(f"   • {issue}")
        
        return report
    
    def run_full_validation(self):
        """Run complete validation suite"""
        print("🚗⚡ EV PROJECT COMPREHENSIVE VALIDATION")
        print("=" * 60)
        
        self.load_data()
        self.validate_data_quality()
        self.validate_business_logic()
        self.validate_model_results()
        report = self.generate_validation_report()
        
        return report

if __name__ == "__main__":
    validator = EVProjectValidator()
    validation_report = validator.run_full_validation()