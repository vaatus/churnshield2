import pandas as pd

# Load the dataset from the file system
user_retention = pd.read_parquet("user_retention.parquet")

# Preview the dataset
print(user_retention.shape)
print(user_retention.dtypes)
user_retention.head()