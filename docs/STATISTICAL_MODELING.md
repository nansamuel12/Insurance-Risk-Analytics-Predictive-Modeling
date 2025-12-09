# Statistical Modeling - Implementation Summary

## Overview
This document provides a comprehensive summary of the Statistical Modeling implementation for the Insurance Risk Analytics Predictive Modeling project.

## Implementation Details

### Module: `src/statistical_modeling.py`

A comprehensive machine learning module that builds, trains, and evaluates multiple regression models to predict insurance charges based on customer risk factors.

## Models Implemented

### 1. Linear Regression
- **Type**: Ordinary Least Squares (OLS) regression
- **Purpose**: Baseline model for comparison
- **Assumptions**: Linear relationship between features and target
- **Pros**: Simple, interpretable, fast training
- **Cons**: May underfit complex relationships
- **Performance**: Test R² = -0.09, RMSE = $6,750

### 2. Ridge Regression (L2 Regularization)
- **Type**: Regularized linear regression
- **Purpose**: Prevent overfitting by penalizing large coefficients
- **Hyperparameter**: Alpha (regularization strength)
- **Tuning**: Grid search with CV over [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
- **Best Alpha**: 10.0
- **Performance**: Test R² = -0.25, RMSE = $7,231

### 3. Lasso Regression (L1 Regularization)
- **Type**: Regularized linear regression with feature selection
- **Purpose**: Automatic feature selection by shrinking coefficients to zero
- **Hyperparameter**: Alpha (regularization strength)
- **Tuning**: Grid search with CV over [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
- **Best Alpha**: 100.0
- **Features Selected**: 6/6 (all features retained)
- **Performance**: Test R² = -0.18, RMSE = $7,002

### 4. Random Forest Regressor
- **Type**: Ensemble of decision trees
- **Purpose**: Capture non-linear relationships and feature interactions
- **Hyperparameters**: 
  - n_estimators: [50, 100]
  - max_depth: [None, 10]
  - min_samples_split: [2, 5]
- **Best Parameters**: Determined via grid search
- **Performance**: Test R² = -0.75, RMSE = $8,538
- **Note**: Severe overfitting (Train R² = 0.90)

### 5. Gradient Boosting Regressor
- **Type**: Sequential ensemble boosting
- **Purpose**: Build strong predictor by combining weak learners
- **Hyperparameters**:
  - n_estimators: [50, 100]
  - learning_rate: [0.1, 0.2]
  - max_depth: [3, 5]
- **Best Parameters**: Determined via grid search
- **Performance**: Test R² = -0.80, RMSE = $8,657
- **Note**: Extreme overfitting (Train R² = 1.00, perfect fit)

## Key Features

### 1. Automated Data Preprocessing
- **Categorical Encoding**: Label encoding for categorical variables (sex, smoker, region)
- **Feature Scaling**: StandardScaler for normalized feature ranges
- **Train/Test Split**: 80/20 split with random_state=42 for reproducibility

### 2. Hyperparameter Optimization
- **Method**: GridSearchCV with cross-validation
- **Cross-Validation**: 3-5 fold CV depending on model
- **Scoring Metric**: Negative mean squared error
- **Search Strategy**: Exhaustive grid search over parameter space

### 3. Comprehensive Evaluation Metrics
- **R² Score**: Proportion of variance explained (1.0 = perfect, 0.0 = baseline, negative = worse than baseline)
- **RMSE**: Root Mean Squared Error (lower is better, in dollars)
- **MAE**: Mean Absolute Error (average prediction error in dollars)
- **MAPE**: Mean Absolute Percentage Error (percentage error)
- **Calculated for**: Both training and test sets to detect overfitting

### 4. Feature Importance Analysis
- **Tree-based Models**: Feature importance from Random Forest and Gradient Boosting
- **Linear Models**: Normalized absolute coefficients from Linear, Ridge, and Lasso
- **Visualization**: Grouped bar chart comparing importance across models

### 5. Model Diagnostics
- **Actual vs Predicted**: Scatter plot with perfect prediction line
- **Residual Plot**: Scatter plot of residuals vs predictions
- **Residual Distribution**: Histogram to check for normality
- **Overfitting Detection**: Automatic flagging when Train R² - Test R² > 0.1

### 6. Comprehensive Reporting
- **JSON Output** (`reports/statistical_modeling_results.json`): Machine-readable results
- **Text Report** (`reports/statistical_modeling_report.txt`): Human-readable detailed report
- **Visualizations**:
  - Model comparison (R², RMSE, MAE, MAPE)
  - Feature importance across models
  - Actual vs predicted scatter plot
  - Residual analysis plots

## Results Summary

### Model Performance Comparison

| Model | Train R² | Test R² | Test RMSE | Test MAE | Test MAPE | Overfitting |
|-------|----------|---------|-----------|----------|-----------|-------------|
| **Linear Regression** | 0.72 | **-0.09** | **$6,750** | **$5,664** | 154% | ⚠️ Yes (0.81) |
| Ridge Regression | 0.61 | -0.25 | $7,231 | $7,220 | 283% | ⚠️ Yes (0.87) |
| Lasso Regression | 0.72 | -0.18 | $7,002 | $5,920 | 162% | ⚠️ Yes (0.90) |
| Random Forest | 0.90 | -0.75 | $8,538 | $7,626 | 273% | 🔴 Severe (1.65) |
| Gradient Boosting | 1.00 | -0.80 | $8,657 | $6,223 | 143% | 🔴 Extreme (1.80) |

**Best Model**: Linear Regression (least overfitting, best test performance)

### Key Findings

#### 1. Small Dataset Challenge
- **Sample Size**: Only 20 records total (16 train, 4 test)
- **Impact**: Insufficient data for reliable model training
- **Consequence**: All models show negative test R², indicating predictions worse than simply using the mean
- **Recommendation**: Collect at least 100-1000 samples for reliable predictions

#### 2. Overfitting Across All Models
- **Linear Models**: Moderate overfitting (R² diff: 0.8-0.9)
- **Tree Models**: Severe overfitting (R² diff: 1.6-1.8)
- **Gradient Boosting**: Perfect training fit (R² = 1.00) but worst test performance
- **Cause**: Model complexity exceeds available training data

#### 3. Model Complexity vs Performance
- **Observation**: More complex models (Random Forest, Gradient Boosting) perform worse
- **Reason**: High-capacity models memorize training data instead of learning patterns
- **Solution**: Use simpler models (Linear Regression) or collect more data

#### 4. Feature Importance Insights
Based on linear model coefficients and tree-based importance scores:
- **Smoking Status**: Most important predictor (consistent with A/B testing results)
- **Age**: Secondary importance
- **BMI**: Moderate importance
- **Children, Sex, Region**: Lower importance

## Technical Implementation

### Code Quality Features
1. **Object-Oriented Design**: `InsuranceRiskModeler` class encapsulates all modeling logic
2. **Type Hints**: Full type annotations for better code clarity
3. **Modular Methods**: Separate methods for each model type
4. **Error Handling**: Robust error handling for data loading and processing
5. **Documentation**: Comprehensive docstrings for all methods
6. **Reproducibility**: Fixed random_state for consistent results
7. **Visualization**: Automated generation of publication-quality plots

### Dependencies
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning models and utilities
- `matplotlib`: Plotting
- `seaborn`: Statistical visualizations

### Integration with DVC Pipeline
The statistical modeling is integrated into the DVC pipeline:
```yaml
statistical_modeling:
  cmd: python src/statistical_modeling.py
  deps:
    - data/raw/data.csv
    - src/statistical_modeling.py
  outs:
    - reports/statistical_modeling_results.json
    - reports/statistical_modeling_report.txt
```

## Usage

### Running the Models
```bash
# Run directly
python src/statistical_modeling.py

# Or through DVC pipeline
dvc repro statistical_modeling
```

### Output Files
1. **reports/statistical_modeling_results.json**: Structured JSON with all model results
2. **reports/statistical_modeling_report.txt**: Detailed text report with interpretations
3. **reports/figures/model_comparison.png**: Model performance comparison charts
4. **reports/figures/feature_importance.png**: Feature importance across models
5. **reports/figures/actual_vs_predicted.png**: Best model predictions vs actual values
6. **reports/figures/residual_analysis.png**: Residual plots for diagnostics

## Interpretation Guidelines

### R² Score
- **1.0**: Perfect predictions
- **0.0**: Model performs as well as predicting the mean
- **Negative**: Model performs worse than predicting the mean
- **Our Results**: All negative, indicating poor generalization

### RMSE (Root Mean Squared Error)
- **Interpretation**: Average prediction error in dollars
- **Lower is better**
- **Our Best**: $6,750 (Linear Regression)
- **Context**: With average charges around $12,000, this is ~56% error

### MAE (Mean Absolute Error)
- **Interpretation**: Average absolute prediction error
- **More robust to outliers than RMSE**
- **Our Best**: $5,664 (Linear Regression)

### MAPE (Mean Absolute Percentage Error)
- **Interpretation**: Percentage error
- **Our Results**: 142-283% (very high)
- **Indicates**: Predictions are highly unreliable

## Recommendations

### Immediate Actions
1. **Data Collection**: Gather significantly more data (target: 500-1000 samples)
2. **Feature Engineering**: Create interaction terms (e.g., smoker × age, smoker × BMI)
3. **Cross-Validation**: Use k-fold CV instead of single train/test split
4. **Regularization**: Increase regularization strength to combat overfitting

### Future Enhancements
1. **Advanced Models**: Try XGBoost, LightGBM, CatBoost
2. **Ensemble Methods**: Combine predictions from multiple models
3. **Feature Selection**: Use recursive feature elimination
4. **Polynomial Features**: Add polynomial terms for non-linear relationships
5. **Outlier Detection**: Identify and handle outliers
6. **Time-Based Validation**: If temporal data, use time-series split
7. **Confidence Intervals**: Add prediction intervals for uncertainty quantification

## Business Implications

### Current State
- **Models Not Production-Ready**: Negative R² scores indicate unreliable predictions
- **Root Cause**: Insufficient training data
- **Risk**: Using these models for pricing would lead to significant errors

### Path Forward
1. **Short-Term**: Use A/B testing results for pricing (smoker vs non-smoker)
2. **Medium-Term**: Collect more data and retrain models
3. **Long-Term**: Build robust ML pipeline with continuous model monitoring

### Value Proposition
Once sufficient data is collected, statistical modeling will enable:
- **Automated Pricing**: Predict charges for new customers
- **Risk Stratification**: Identify high-risk vs low-risk customers
- **What-If Analysis**: Simulate impact of policy changes
- **Personalization**: Tailor offerings based on customer profiles

## Conclusion

The Statistical Modeling implementation provides a robust, automated framework for building and evaluating predictive models. While current results show overfitting due to limited data, the infrastructure is production-ready and will deliver accurate predictions once more data is collected. The modular design allows easy addition of new models and features as the project evolves.

---
**Author**: nansamuel12  
**Date**: December 2025  
**Project**: Insurance Risk Analytics Predictive Modeling
