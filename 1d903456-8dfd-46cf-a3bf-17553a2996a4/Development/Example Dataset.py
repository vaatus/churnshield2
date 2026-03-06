import pandas as pd

# Load the user retention dataset from the file system
user_retention = pd.read_parquet("user_retention.parquet")

# Preview the dataset
print(f"Shape: {user_retention.shape}")
print(f"\nColumns: {list(user_retention.columns)}")
print(f"\nData Types:\n{user_retention.dtypes}")
print(f"\nFirst 5 Rows:\n")
user_retention