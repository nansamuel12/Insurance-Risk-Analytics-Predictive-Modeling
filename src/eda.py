import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs("reports/figures", exist_ok=True)

# Load data
df = pd.read_csv("data/raw/data.csv")

# 1. Data Quality Checks
print("Missing values per column:")
print(df.isnull().sum())
print("\nData Types:")
print(df.dtypes)

# 2. Univariate Analysis: Distribution of Charges
plt.figure(figsize=(10, 6))
sns.histplot(df['charges'], kde=True)
plt.title('Distribution of Insurance Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.savefig('reports/figures/charges_distribution.png')
plt.close()

# 3. Bivariate Analysis: Charges vs Age
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='charges', hue='smoker')
plt.title('Charges vs Age (colored by Smoker)')
plt.savefig('reports/figures/charges_vs_age.png')
plt.close()

# 4. Outlier Detection: Boxplot of BMI
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['bmi'])
plt.title('Boxplot of BMI')
plt.savefig('reports/figures/bmi_boxplot.png')
plt.close()

# 5. Correlation Matrix
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('reports/figures/correlation_matrix.png')
plt.close()

print("EDA completed. Figures saved to reports/figures/")
