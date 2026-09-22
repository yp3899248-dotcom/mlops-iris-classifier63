import pandas as pd
from datetime import datetime, timezone

input_path = "../../data/processed/iris_features.csv"
output_path = "data/iris_features.parquet"

df = pd.read_csv(input_path)

# Add Feast entity key
df["sample_id"] = range(1, len(df) + 1)

# Add timestamps
event_time = datetime(2026, 8, 15, 15, 20, 2, tzinfo=timezone.utc)
df["event_timestamp"] = event_time
df["created_timestamp"] = event_time

# Save as Parquet
df.to_parquet(output_path, index=False)

print(f"Saved {output_path}")
print(f"Shape: {df.shape}")
print(df.head())