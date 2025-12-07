import pandas as pd
import json
import os

# Read processed data
df = pd.read_csv("data/processed/processed_data.csv")

# Simple "training" / analysis: Calculate mean and std value
mean_val = df['value'].mean()
std_val = df['value'].std()

# Save metrics
metrics = {
    "mean_value": mean_val,
    "std_value": std_val
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Model trained. Metrics saved: {metrics}")
