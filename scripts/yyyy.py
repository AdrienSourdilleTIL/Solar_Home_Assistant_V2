import pandas as pd
import os
from glob import glob

# Folder containing yearly CSVs
data_folder = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\consumption\raw"

# Match all CSVs like 2015.csv, 2016.csv, ...
file_pattern = os.path.join(data_folder, "[0-9][0-9][0-9][0-9].csv")
csv_files = sorted(glob(file_pattern))

# Read and concatenate all
dfs = [pd.read_csv(file) for file in csv_files]
full_df = pd.concat(dfs, ignore_index=True)

# Save to one big CSV
output_file = os.path.join(data_folder, "2015_2023_hourly.csv")
full_df.to_csv(output_file, index=False)
print(f"Combined CSV saved to {output_file}")
