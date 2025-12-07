import pandas as pd
import os

# Ensure output directory exists
os.makedirs("data/processed", exist_ok=True)

# Read data
df = pd.read_csv("data/raw/data.csv")

# Process data: Double the charges (dummy transformation)
df['charges'] = df['charges'] * 2

# Save processed data
df.to_csv("data/processed/processed_data.csv", index=False)
print("Data processed and saved to data/processed/processed_data.csv")
