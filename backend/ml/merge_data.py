# merge_data.py
import pandas as pd
import glob

files = glob.glob("dataset/*.csv")
dfs = [pd.read_csv(f) for f in files]

df = pd.concat(dfs, ignore_index=True)
df.to_csv("dataset/train.csv", index=False)
print("✅ Merged dataset saved as train.csv")