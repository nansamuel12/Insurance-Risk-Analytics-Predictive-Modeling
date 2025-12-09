"""
Statistical Modeling Module for Insurance Risk Analytics

This module implements comprehensive statistical and machine learning models
to predict insurance charges based on various risk factors. It includes:
- Linear Regression
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Random Forest Regression
- Gradient Boosting Regression
- Model evaluation and comparison
- Feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, List, Tuple, Any
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')


class InsuranceRiskModeler:
    """
    Statistical modeling class for insurance risk analytics.
    
    This class implements multiple regression models to predict insurance
    charges and evaluates their performance using various metrics.
    """
    
    def __init__(self, data: pd.DataFrame, target_column: str = 'charges', 
                 test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the Insurance Risk Modeler.
        
        Args:
            data (pd.DataFrame): Insurance dataset
            target_column (str): Name of the target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        self.data = data.copy()
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
        self.models = {}
        self.results = {}
        self.predictions = {}
        
    def prepare_data(self) -> None:
        """
        Prepare data for modeling by encoding categorical variables
        and splitting into train/test sets.
        """
        print("Preparing data for modeling...")
        
        # Separate features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        # Encode categorical variables
        label_encoders = {}
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"[OK] Data prepared: {len(self.X_train)} training samples, "
              f"{len(self.X_test)} test samples")
        print(f"[OK] Features: {', '.join(self.feature_names)}")
    
    def train_linear_regression(self) -> Dict[str, Any]:
        """
        Train a Linear Regression model.
        
        Returns:
            Dict: Model performance metrics
        """
        print("\nTraining Linear Regression...")
        
        model = LinearRegression()
        model.fit(self.X_train_scaled, self.y_train)
        
        self.models['Linear Regression'] = model
        metrics = self._evaluate_model(model, 'Linear Regression', 
                                       self.X_train_scaled, self.X_test_scaled)
        
        print(f"[OK] Linear Regression - R2: {metrics['test_r2']:.4f}, "
              f"RMSE: ${metrics['test_rmse']:.2f}")
        
        return metrics
    
    def train_ridge_regression(self, alpha: float = 1.0) -> Dict[str, Any]:
        """
        Train a Ridge Regression model (L2 regularization).
        
        Args:
            alpha (float): Regularization strength
            
        Returns:
            Dict: Model performance metrics
        """
        print("\nTraining Ridge Regression...")
        
        # Grid search for best alpha
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        ridge = Ridge(random_state=self.random_state)
        
        grid_search = GridSearchCV(
            ridge, param_grid, cv=5, 
            scoring='neg_mean_squared_error', n_jobs=1
        )
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        best_model = grid_search.best_estimator_
        self.models['Ridge Regression'] = best_model
        
        metrics = self._evaluate_model(best_model, 'Ridge Regression',
                                       self.X_train_scaled, self.X_test_scaled)
        metrics['best_alpha'] = grid_search.best_params_['alpha']
        
        print(f"[OK] Ridge Regression - Best alpha: {metrics['best_alpha']}, "
              f"R2: {metrics['test_r2']:.4f}, RMSE: ${metrics['test_rmse']:.2f}")
        
        return metrics
    
    def train_lasso_regression(self, alpha: float = 1.0) -> Dict[str, Any]:
        """
        Train a Lasso Regression model (L1 regularization).
        
        Args:
            alpha (float): Regularization strength
            
        Returns:
            Dict: Model performance metrics
        """
        print("\nTraining Lasso Regression...")
        
        # Grid search for best alpha
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        lasso = Lasso(random_state=self.random_state, max_iter=10000)
        
        grid_search = GridSearchCV(
            lasso, param_grid, cv=5,
            scoring='neg_mean_squared_error', n_jobs=1
        )
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        best_model = grid_search.best_estimator_
        self.models['Lasso Regression'] = best_model
        
        metrics = self._evaluate_model(best_model, 'Lasso Regression',
                                       self.X_train_scaled, self.X_test_scaled)
        metrics['best_alpha'] = grid_search.best_params_['alpha']
        
        # Feature selection (non-zero coefficients)
        non_zero_features = np.sum(best_model.coef_ != 0)
        metrics['selected_features'] = int(non_zero_features)
        
        print(f"[OK] Lasso Regression - Best alpha: {metrics['best_alpha']}, "
              f"R2: {metrics['test_r2']:.4f}, RMSE: ${metrics['test_rmse']:.2f}, "
              f"Features selected: {non_zero_features}/{len(self.feature_names)}")
        
        return metrics
    
    def train_random_forest(self) -> Dict[str, Any]:
        """
        Train a Random Forest Regression model.
        
        Returns:
            Dict: Model performance metrics
        """
        print("\nTraining Random Forest Regressor...")
        
        # Simplified grid for small datasets
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=3,
            scoring='neg_mean_squared_error', n_jobs=1, verbose=0
        )
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        self.models['Random Forest'] = best_model
        
        metrics = self._evaluate_model(best_model, 'Random Forest',
                                       self.X_train, self.X_test)
        metrics['best_params'] = grid_search.best_params_
        
        print(f"[OK] Random Forest - R2: {metrics['test_r2']:.4f}, "
              f"RMSE: ${metrics['test_rmse']:.2f}")
        
        return metrics
    
    def train_gradient_boosting(self) -> Dict[str, Any]:
        """
        Train a Gradient Boosting Regression model.
        
        Returns:
            Dict: Model performance metrics
        """
        print("\nTraining Gradient Boosting Regressor...")
        
        # Simplified grid for small datasets
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 5]
        }
        
        gb = GradientBoostingRegressor(random_state=self.random_state)
        
        grid_search = GridSearchCV(
            gb, param_grid, cv=3,
            scoring='neg_mean_squared_error', n_jobs=1, verbose=0
        )
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        self.models['Gradient Boosting'] = best_model
        
        metrics = self._evaluate_model(best_model, 'Gradient Boosting',
                                       self.X_train, self.X_test)
        metrics['best_params'] = grid_search.best_params_
        
        print(f"[OK] Gradient Boosting - R2: {metrics['test_r2']:.4f}, "
              f"RMSE: ${metrics['test_rmse']:.2f}")
        
        return metrics
    
    def _evaluate_model(self, model: Any, model_name: str,
                       X_train: np.ndarray, X_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a trained model on both training and test sets.
        
        Args:
            model: Trained model
            model_name (str): Name of the model
            X_train: Training features
            X_test: Test features
            
        Returns:
            Dict: Performance metrics
        """
        # Training predictions
        y_train_pred = model.predict(X_train)
        
        # Test predictions
        y_test_pred = model.predict(X_test)
        
        # Store predictions
        self.predictions[model_name] = {
            'train': y_train_pred,
            'test': y_test_pred
        }
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            
            # Training metrics
            'train_r2': r2_score(self.y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'train_mape': mean_absolute_percentage_error(self.y_train, y_train_pred) * 100,
            
            # Test metrics
            'test_r2': r2_score(self.y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred),
            'test_mape': mean_absolute_percentage_error(self.y_test, y_test_pred) * 100,
        }
        
        self.results[model_name] = metrics
        
        return metrics
    
    def train_all_models(self) -> Dict[str, Dict[str, float]]:
        """
        Train all available models.
        
        Returns:
            Dict: All model results
        """
        print("=" * 80)
        print("TRAINING ALL STATISTICAL MODELS")
        print("=" * 80)
        
        self.prepare_data()
        
        # Train all models
        self.train_linear_regression()
        self.train_ridge_regression()
        self.train_lasso_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        
        return self.results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        importance_data = []
        
        # Random Forest
        if 'Random Forest' in self.models:
            rf_importance = self.models['Random Forest'].feature_importances_
            for feat, imp in zip(self.feature_names, rf_importance):
                importance_data.append({
                    'model': 'Random Forest',
                    'feature': feat,
                    'importance': imp
                })
        
        # Gradient Boosting
        if 'Gradient Boosting' in self.models:
            gb_importance = self.models['Gradient Boosting'].feature_importances_
            for feat, imp in zip(self.feature_names, gb_importance):
                importance_data.append({
                    'model': 'Gradient Boosting',
                    'feature': feat,
                    'importance': imp
                })
        
        # Linear models (absolute coefficients)
        for model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            if model_name in self.models:
                coef = np.abs(self.models[model_name].coef_)
                # Normalize to sum to 1
                coef_normalized = coef / coef.sum() if coef.sum() > 0 else coef
                for feat, imp in zip(self.feature_names, coef_normalized):
                    importance_data.append({
                        'model': model_name,
                        'feature': feat,
                        'importance': imp
                    })
        
        return pd.DataFrame(importance_data)
    
    def compare_models(self) -> pd.DataFrame:
        """
        Create a comparison table of all models.
        
        Returns:
            pd.DataFrame: Model comparison table
        """
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train R²': metrics['train_r2'],
                'Test R²': metrics['test_r2'],
                'Train RMSE': metrics['train_rmse'],
                'Test RMSE': metrics['test_rmse'],
                'Train MAE': metrics['train_mae'],
                'Test MAE': metrics['test_mae'],
                'Test MAPE (%)': metrics['test_mape']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test R²', ascending=False)
        
        return df
    
    def visualize_results(self, output_dir: str = "reports/figures") -> None:
        """
        Create visualizations for model results.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Model Comparison - R² Scores
        print("\nGenerating visualizations...")
        
        comparison_df = self.compare_models()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # R² Comparison
        ax = axes[0, 0]
        x = np.arange(len(comparison_df))
        width = 0.35
        ax.bar(x - width/2, comparison_df['Train R²'], width, label='Train R²', alpha=0.8)
        ax.bar(x + width/2, comparison_df['Test R²'], width, label='Test R²', alpha=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel('R² Score')
        ax.set_title('Model Comparison: R² Scores')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # RMSE Comparison
        ax = axes[0, 1]
        ax.bar(x - width/2, comparison_df['Train RMSE'], width, label='Train RMSE', alpha=0.8)
        ax.bar(x + width/2, comparison_df['Test RMSE'], width, label='Test RMSE', alpha=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel('RMSE ($)')
        ax.set_title('Model Comparison: RMSE')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # MAE Comparison
        ax = axes[1, 0]
        ax.bar(x - width/2, comparison_df['Train MAE'], width, label='Train MAE', alpha=0.8)
        ax.bar(x + width/2, comparison_df['Test MAE'], width, label='Test MAE', alpha=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel('MAE ($)')
        ax.set_title('Model Comparison: Mean Absolute Error')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # MAPE Comparison
        ax = axes[1, 1]
        ax.bar(comparison_df['Model'], comparison_df['Test MAPE (%)'], alpha=0.8, color='coral')
        ax.set_xlabel('Model')
        ax.set_ylabel('MAPE (%)')
        ax.set_title('Model Comparison: Mean Absolute Percentage Error')
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("[OK] Saved model comparison visualization")
        
        # 2. Feature Importance
        importance_df = self.get_feature_importance()
        
        if not importance_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Pivot for grouped bar chart
            pivot_df = importance_df.pivot(index='feature', columns='model', values='importance')
            pivot_df.plot(kind='bar', ax=ax, alpha=0.8)
            
            ax.set_xlabel('Feature')
            ax.set_ylabel('Importance')
            ax.set_title('Feature Importance Across Models')
            ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            print("[OK] Saved feature importance visualization")
        
        # 3. Actual vs Predicted (Best Model)
        best_model_name = comparison_df.iloc[0]['Model']
        best_predictions = self.predictions[best_model_name]['test']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.y_test, best_predictions, alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), best_predictions.min())
        max_val = max(self.y_test.max(), best_predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Charges ($)')
        ax.set_ylabel('Predicted Charges ($)')
        ax.set_title(f'Actual vs Predicted: {best_model_name}\n'
                    f'R² = {self.results[best_model_name]["test_r2"]:.4f}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("[OK] Saved actual vs predicted visualization")
        
        # 4. Residual Plot (Best Model)
        residuals = self.y_test - best_predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Residual scatter plot
        ax = axes[0]
        ax.scatter(best_predictions, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Charges ($)')
        ax.set_ylabel('Residuals ($)')
        ax.set_title(f'Residual Plot: {best_model_name}')
        ax.grid(alpha=0.3)
        
        # Residual distribution
        ax = axes[1]
        ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Residuals ($)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Residual Distribution: {best_model_name}')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'residual_analysis.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("[OK] Saved residual analysis visualization")
    
    def save_results(self, output_path: str = "reports/statistical_modeling_results.json") -> None:
        """
        Save all modeling results to a JSON file.
        
        Args:
            output_path (str): Path to save results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy types to Python types
        results_serializable = {}
        for model_name, metrics in self.results.items():
            results_serializable[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
                if k not in ['best_params']  # Skip complex objects
            }
        
        output_data = {
            'metadata': {
                'target_variable': self.target_column,
                'test_size': self.test_size,
                'random_state': self.random_state,
                'n_train_samples': len(self.X_train),
                'n_test_samples': len(self.X_test),
                'n_features': len(self.feature_names),
                'features': self.feature_names
            },
            'results': results_serializable,
            'best_model': self.compare_models().iloc[0]['Model']
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"\n[OK] Results saved to {output_path}")
    
    def generate_report(self, output_path: str = "reports/statistical_modeling_report.txt") -> None:
        """
        Generate a comprehensive text report of all models.
        
        Args:
            output_path (str): Path to save report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        comparison_df = self.compare_models()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("STATISTICAL MODELING REPORT\n")
            f.write("Insurance Risk Analytics - Predictive Modeling\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Target Variable: {self.target_column}\n")
            f.write(f"Training Samples: {len(self.X_train)}\n")
            f.write(f"Test Samples: {len(self.X_test)}\n")
            f.write(f"Features: {', '.join(self.feature_names)}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("MODEL COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("BEST MODEL\n")
            f.write("=" * 80 + "\n\n")
            
            best_model_name = comparison_df.iloc[0]['Model']
            best_metrics = self.results[best_model_name]
            
            f.write(f"Model: {best_model_name}\n")
            f.write(f"Test R² Score: {best_metrics['test_r2']:.4f}\n")
            f.write(f"Test RMSE: ${best_metrics['test_rmse']:.2f}\n")
            f.write(f"Test MAE: ${best_metrics['test_mae']:.2f}\n")
            f.write(f"Test MAPE: {best_metrics['test_mape']:.2f}%\n\n")
            
            f.write("Interpretation:\n")
            f.write(f"The {best_model_name} model explains {best_metrics['test_r2']*100:.2f}% ")
            f.write(f"of the variance in insurance charges.\n")
            f.write(f"On average, predictions are off by ${best_metrics['test_mae']:.2f} ")
            f.write(f"({best_metrics['test_mape']:.2f}%).\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("DETAILED MODEL RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Training Performance:\n")
                f.write(f"  R² Score: {metrics['train_r2']:.4f}\n")
                f.write(f"  RMSE: ${metrics['train_rmse']:.2f}\n")
                f.write(f"  MAE: ${metrics['train_mae']:.2f}\n")
                f.write(f"  MAPE: {metrics['train_mape']:.2f}%\n\n")
                
                f.write(f"Test Performance:\n")
                f.write(f"  R² Score: {metrics['test_r2']:.4f}\n")
                f.write(f"  RMSE: ${metrics['test_rmse']:.2f}\n")
                f.write(f"  MAE: ${metrics['test_mae']:.2f}\n")
                f.write(f"  MAPE: {metrics['test_mape']:.2f}%\n\n")
                
                # Check for overfitting
                r2_diff = metrics['train_r2'] - metrics['test_r2']
                if r2_diff > 0.1:
                    f.write(f"  Note: Possible overfitting detected (R² difference: {r2_diff:.4f})\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"[OK] Report saved to {output_path}")


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at {filepath}")
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def main():
    """Main execution function."""
    print("=" * 80)
    print("STATISTICAL MODELING - Insurance Risk Analytics")
    print("=" * 80)
    
    # Load data
    input_path = "data/raw/data.csv"
    print(f"\nLoading data from {input_path}...")
    df = load_data(input_path)
    print(f"[OK] Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Initialize modeler
    modeler = InsuranceRiskModeler(df, target_column='charges', test_size=0.2)
    
    # Train all models
    results = modeler.train_all_models()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    modeler.visualize_results()
    
    # Save results
    print("\nSaving results...")
    modeler.save_results()
    modeler.generate_report()
    
    # Print summary
    print("\n" + "=" * 80)
    print("STATISTICAL MODELING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    comparison_df = modeler.compare_models()
    print("\nModel Performance Summary:")
    print(comparison_df.to_string(index=False))
    
    print(f"\nBest Model: {comparison_df.iloc[0]['Model']}")
    print(f"Test R² Score: {comparison_df.iloc[0]['Test R²']:.4f}")
    print(f"Test RMSE: ${comparison_df.iloc[0]['Test RMSE']:.2f}")
    print("\nCheck reports/ directory for detailed results and visualizations")


if __name__ == "__main__":
    main()
