import pandas as pd
# df = pd.read_parquet("hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet")

df = pd.read_parquet("hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet")
print(df.head())