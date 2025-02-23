import os
import pandas as pd

# Define the base directory
docdir = '/mwtx0006-nas01.bankofamerica.com/dcrs_acde$/POC/Gen_AI/ESO_Proj_May2025/Documents/UAT_groundtruth_DIVO/Death-Certificate/texts'

print(f"Processing directory: {docdir}")

csv_files = []  # List to store CSV file paths

# List of all channels to process
channels = ['EML', 'FAX', 'WIN', 'SCN']

# Iterate through each channel
for subfolder in channels:
    path = os.path.join(docdir, subfolder)

    # Check if the path is a directory
    if not os.path.isdir(path):
        print(f"Skipping {path}, not a directory")
        continue

    # List all items in the subfolder
    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        # Check if the item is a file and ends with .csv
        if os.path.isfile(item_path) and item_path.endswith('.csv'):
            csv_files.append(item_path)
        else:
            print(f"Skipping {item_path}, not a CSV file")

# Print all found CSV files
print("\nFound CSV files:")
for csv in csv_files:
    print(csv)

# Process CSV files
data = []  # Store dataframes

for file_path in csv_files:
    try:
        df = pd.read_csv(file_path, header=None, names=['text'])  # Read CSV without header
        df['PE_num'] = range(1, len(df) + 1)  # Assign page numbers
        df['fn'] = os.path.basename(file_path)  # Assign filename
        df['channel'] = os.path.basename(os.path.dirname(file_path))  # Assign channel name
        df['fp'] = file_path  # Store the full file path
        data.append(df)  # Append to list

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Concatenate all data into a single DataFrame
if data:
    combined_df = pd.concat(data, ignore_index=True)

    # Save to a new CSV file
    output_file = '/mwtx0006-nas01.bankofamerica.com/dcrs_acde$/POC/Gen_AI/ESO_Proj_May2025/Documents/texts/All_Channels_Data.csv'
    combined_df.to_csv(output_file, index=False)

    print(f"\nData saved to: {output_file}")
else:
    print("\nNo data was processed.")
