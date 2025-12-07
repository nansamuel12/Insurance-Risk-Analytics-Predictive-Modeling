import pandas as pd
import os
import sys

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
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

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the dataframe.
    
    Doubles the 'charges' column as a dummy transformation.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Processed dataframe.
    """
    if 'charges' not in df.columns:
        print("Error: 'charges' column missing from input data.")
        sys.exit(1)
    
    df = df.copy()
    try:
        df['charges'] = df['charges'] * 2
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)
        
    return df

def save_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save dataframe to a CSV file.
    
    Args:
        df (pd.DataFrame): Dataframe to save.
        filepath (str): Destination path.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Data processed and saved to {filepath}")
    except Exception as e:
        print(f"Error saving data: {e}")
        sys.exit(1)

def main():
    """Main execution function."""
    input_path = "data/raw/data.csv"
    output_path = "data/processed/processed_data.csv"
    
    print("Starting data processing...")
    df = load_data(input_path)
    processed_df = process_data(df)
    save_data(processed_df, output_path)
    print("Data processing completed successfully.")

if __name__ == "__main__":
    main()
