# Electric Vehicle Analysis - Machine Learning Project

## Project Overview

This is a comprehensive machine learning project analyzing electric vehicle specifications and performance metrics. The project demonstrates end-to-end data science workflow from data exploration to model deployment.

### Dataset Information
- **Records**: 478 electric vehicles
- **Features**: 22 specifications including battery capacity, range, efficiency, performance metrics
- **Source**: Electric vehicle specifications dataset (2025)

## Project Structure

```
ev-ml-project/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned and processed data
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Feature creation and preprocessing
│   ├── 03_model_development.ipynb    # Model training and tuning
│   └── 04_model_evaluation.ipynb     # Model evaluation and comparison
├── src/
│   ├── data/                   # Data processing modules
│   ├── features/               # Feature engineering
│   ├── models/                 # Model training and prediction
│   ├── visualization/          # Plotting and visualization
│   └── utils/                  # Utility functions
├── dashboard/
│   └── app.py                  # Interactive Streamlit dashboard
├── reports/
│   ├── figures/                # Generated plots and charts
│   └── html_reports/           # Notebook HTML exports
├── models/                     # Saved trained models
├── tests/                      # Unit tests
└── config/                     # Configuration files
```

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd ev-ml-project

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Analysis
```bash
# Execute full automation pipeline
python run_project_fixed.py
```

### 3. Launch Dashboard
```bash
# Start interactive dashboard
streamlit run dashboard/app.py
```

## Analysis Results

### Key Insights
- Average electric vehicle range: Analyzed across different segments
- Battery efficiency patterns: Relationship between capacity and real-world performance
- Market segmentation: Classification of vehicles by use case and performance
- Performance factors: Key drivers of acceleration and top speed

### Machine Learning Models

#### 1. Range Prediction (Regression)
- **Objective**: Predict vehicle range based on specifications
- **Models**: Linear Regression, Random Forest, XGBoost, LightGBM
- **Best Performance**: [Results populated after execution]

#### 2. Segment Classification
- **Objective**: Classify vehicles into market segments
- **Models**: Logistic Regression, SVM, Random Forest, XGBoost
- **Accuracy**: [Results populated after execution]

#### 3. Vehicle Clustering
- **Objective**: Group similar vehicles for market analysis
- **Methods**: K-Means, DBSCAN, Hierarchical clustering
- **Insights**: [Results populated after execution]

## Dashboard Features

The interactive Streamlit dashboard provides:

- **Data Explorer**: Interactive data visualization and filtering
- **Model Performance**: Comparison of different ML algorithms
- **Prediction Interface**: Input vehicle specs for range/segment prediction
- **Market Insights**: Business intelligence and trend analysis
- **Model Explanations**: SHAP values and feature importance

## Technical Implementation

### Data Processing
- Automated data validation and quality checks
- Missing value imputation strategies
- Feature scaling and normalization
- Categorical encoding (one-hot, label, target encoding)

### Model Development
- Cross-validation with multiple metrics
- Hyperparameter tuning using GridSearchCV
- Model selection based on business requirements
- Performance monitoring and validation

### Production Features
- Modular code architecture with OOP design
- Comprehensive error handling and logging
- Unit tests for core functionality
- Configuration management
- Model versioning and experiment tracking

## Business Applications

### For Consumers
- **Range Prediction**: Estimate real-world vehicle range
- **Vehicle Comparison**: Compare specs across different models
- **Purchase Decision**: Data-driven vehicle selection

### For Manufacturers
- **Market Analysis**: Understand competitive landscape
- **Product Development**: Identify optimization opportunities
- **Pricing Strategy**: Position vehicles based on performance metrics

### For Researchers
- **Technology Trends**: Analyze battery efficiency improvements
- **Performance Analysis**: Understand engineering trade-offs
- **Market Evolution**: Track industry developments

## API Endpoints

The project includes FastAPI endpoints for production use:

```python
# Predict vehicle range
POST /predict/range
{
    "battery_capacity_kWh": 75.0,
    "efficiency_wh_per_km": 180,
    "drivetrain": "AWD"
}

# Classify vehicle segment
POST /predict/segment
{
    "top_speed_kmh": 200,
    "acceleration_0_100_s": 6.5,
    "price_range": "premium"
}
```

## Model Performance

### Regression Models (Range Prediction)
| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | [TBD] | [TBD] | [TBD] |
| Random Forest | [TBD] | [TBD] | [TBD] |
| XGBoost | [TBD] | [TBD] | [TBD] |
| LightGBM | [TBD] | [TBD] | [TBD] |

### Classification Models (Segment Prediction)
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | [TBD] | [TBD] | [TBD] | [TBD] |
| Random Forest | [TBD] | [TBD] | [TBD] | [TBD] |
| XGBoost | [TBD] | [TBD] | [TBD] | [TBD] |
| SVM | [TBD] | [TBD] | [TBD] | [TBD] |

## Deployment

### Local Deployment
```bash
# Run full pipeline
python run_project_fixed.py

# Launch dashboard
streamlit run dashboard/app.py

# Start API server
uvicorn src.api.main:app --reload
```

### Docker Deployment
```bash
# Build container
docker build -t ev-ml-project .

# Run container
docker run -p 8501:8501 ev-ml-project
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Run specific test suite
python -m pytest tests/test_models.py

# Generate coverage report
python -m pytest --cov=src tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: [Credit data source]
- Scikit-learn community for ML algorithms
- Streamlit team for dashboard framework
- Open source contributors

## Contact

For questions or collaboration opportunities:
- Email: [your-email]
- LinkedIn: [your-linkedin]
- GitHub: [your-github]

---

**Generated on**: 2025-07-12 10:39:43
**Project Status**: Ready for production deployment
**Last Updated**: 2025-07-12
