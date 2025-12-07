import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV."""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at {filepath}")
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def check_data_quality(df: pd.DataFrame) -> None:
    """Print missing values and data types."""
    print("Missing values per column:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)

def plot_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Plot distribution of charges."""
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['charges'], kde=True)
        plt.title('Distribution of Insurance Charges')
        plt.xlabel('Charges')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'charges_distribution.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting distribution: {e}")

def plot_bivariate(df: pd.DataFrame, output_dir: str) -> None:
    """Plot charges vs age."""
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='age', y='charges', hue='smoker')
        plt.title('Charges vs Age (colored by Smoker)')
        plt.savefig(os.path.join(output_dir, 'charges_vs_age.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting bivariate analysis: {e}")

def plot_outliers(df: pd.DataFrame, output_dir: str) -> None:
    """Plot BMI boxplot."""
    try:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df['bmi'])
        plt.title('Boxplot of BMI')
        plt.savefig(os.path.join(output_dir, 'bmi_boxplot.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting outliers: {e}")

def plot_correlation(df: pd.DataFrame, output_dir: str) -> None:
    """Plot correlation matrix."""
    try:
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting correlation: {e}")

def main():
    """Main execution function."""
    input_path = "data/raw/data.csv"
    output_dir = "reports/figures"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting EDA...")
    df = load_data(input_path)
    
    check_data_quality(df)
    plot_distribution(df, output_dir)
    plot_bivariate(df, output_dir)
    plot_outliers(df, output_dir)
    plot_correlation(df, output_dir)
    
    print(f"EDA completed. Figures saved to {output_dir}/")

if __name__ == "__main__":
    main()
