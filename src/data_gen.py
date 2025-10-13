import os
import pandas as pd
import numpy as np

# Define output folder and file name
output_dir = os.path.join("data", "test")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "sample_upload.csv")

# Create fake test data
np.random.seed(42)
data = {
    "sample_id": [f"S{i}" for i in range(1, 11)],
    "temperature": np.random.uniform(15, 35, 10).round(2),
    "humidity": np.random.uniform(30, 90, 10).round(2),
    "pollution": np.random.uniform(10, 100, 10).round(2),
    "microbe_count": np.random.randint(100, 10000, 10)
}

# Write CSV
df = pd.DataFrame(data)
df.to_csv(output_file, index=False)

print(f"Wrote {os.path.abspath(output_file)}")
