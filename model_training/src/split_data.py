from GLOBS import DATA_SLUG
import csv
import random

# Load all data from the cleaned dataset
file = []

with open(f"{DATA_SLUG}cleaned_cbb.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        file.append(row)

num_glob_rows = len(file)

# Shuffle the data using Fisher-Yates algorithm
# Start from index 1 to preserve the header row at index 0
for r in range(1, num_glob_rows):
    swap_idx = random.randint(1, num_glob_rows - 1)
    temp_row = file[r]
    file[r] = file[swap_idx]
    file[swap_idx] = temp_row

# Extract header row
glob_header = file[0]

# Split data into training (33%), validation (33%), and test (33%) sets
training_file = file[1 : num_glob_rows // 3]
val_file = file[num_glob_rows // 3 : 2 * (num_glob_rows // 3)]
test_file = file[2 * (num_glob_rows // 3) :]

# Write training data to file
with open(f"{DATA_SLUG}training_data.csv", "x") as f:
    writer = csv.writer(f)
    writer.writerow(glob_header)
    for row in training_file:
        writer.writerow(row)

# Write validation data to file
with open(f"{DATA_SLUG}val_data.csv", "x") as f:
    writer = csv.writer(f)
    writer.writerow(glob_header)
    for row in val_file:
        writer.writerow(row)

# Write test data to file
with open(f"{DATA_SLUG}test_data.csv", "x") as f:
    writer = csv.writer(f)
    writer.writerow(glob_header)
    for row in test_file:
        writer.writerow(row)
