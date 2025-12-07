import pandas as pd
import json
import os

# Read processed data
df = pd.read_csv("data/processed/processed_data.csv")

# Simple "training" / analysis: Calculate mean value
mean_val = df['value'].mean()

# Save metrics
metrics = {"mean_value": mean_val}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"Model trained. Metrics saved: {metrics}")
