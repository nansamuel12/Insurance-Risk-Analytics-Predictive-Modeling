import pandas as pd
import json
import sys
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load processed data from CSV.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at {filepath}")
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate mean and standard deviation of charges.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        dict: Dictionary containing calculated metrics.
    """
    if 'charges' not in df.columns:
        print("Error: 'charges' column missing.")
        sys.exit(1)
        
    try:
        return {
            "mean_value": float(df['charges'].mean()),
            "std_value": float(df['charges'].std())
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        sys.exit(1)

def save_metrics(metrics: dict, filepath: str) -> None:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics (dict): Metrics to save.
        filepath (str): Destination path.
    """
    try:
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Model trained. Metrics saved: {metrics}")
    except Exception as e:
        print(f"Error saving metrics: {e}")
        sys.exit(1)

def main():
    """Main execution function."""
    input_path = "data/processed/processed_data.csv"
    metrics_path = "metrics.json"
    
    print("Starting model training/analysis...")
    df = load_data(input_path)
    metrics = calculate_metrics(df)
    save_metrics(metrics, metrics_path)
    print("Model training completed successfully.")

if __name__ == "__main__":
    main()
