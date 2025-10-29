import pandas as pd
df = pd.read_csv("data/mutual-fund-data.csv")
print(df.columns.tolist())
print(df.head())
