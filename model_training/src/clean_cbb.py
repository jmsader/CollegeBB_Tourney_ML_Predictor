import csv
import numpy as np
from GLOBS import DATA_SLUG

# Read the original basketball data file
orig_file = []
with open(f"{DATA_SLUG}cbb.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        orig_file.append(row)

# Map tournament results to numeric values for model training
# Higher numbers represent better tournament performance
result_map = {}
result_map["NA"] = 0  # Not applicable/no tournament appearance
result_map["N/A"] = 0  # Not applicable/no tournament appearance
result_map["R68"] = 1  # First Four (Round of 68)
result_map["R64"] = 2  # Round of 64
result_map["R32"] = 3  # Round of 32
result_map["S16"] = 4  # Sweet 16
result_map["E8"] = 5  # Elite 8
result_map["F4"] = 6  # Final Four
result_map["2ND"] = 7  # Runner-up (2nd place)
result_map["Champions"] = 8  # Tournament champions

# Convert tournament result strings to numeric values
# Skip header row (index 0) and process all data rows
for r in range(1, len(orig_file)):
    orig_file[r][-3] = result_map[str(orig_file[r][-3])]

# Convert to numpy array for easier manipulation
orig_file = np.array(orig_file)
# Remove the second-to-last column
orig_file = np.delete(orig_file, -2, 1)

# Write the cleaned data to a new CSV file
# Keep only columns from index 4 onwards
with open(f"{DATA_SLUG}cleaned_cbb.csv", "x") as f:
    writer = csv.writer(f)
    for row in orig_file:
        writer.writerow(row[4:])
