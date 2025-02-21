import os
import pandas as pd

# Initialize empty lists and data container
csv_files_l = list()
csv_fp_l = list()
csv_channel_l = list()
data = []

# Set the directory path
docdir = "//mwtx0006-nas01.bankofamerica.com/dcrs_acde$/POC/Gen_AI/ESO_Proj_May2025/Documents/texts/"
print(docdir)

# For each channel (using 'EML' as shown in your code)
for ch in ["EML"]:
    # Path to the channel
    path = os.path.join(docdir, ch)
    print(path)
    
    # Get all CSV files directly in the channel folder
    csv_files = [f for f in os.listdir(path) if f.endswith('.pdf')]  # Note: Changed to .pdf based on your image
    
    # Process each file
    for file in csv_files:
        file_path = os.path.join(path, file)
        csv_fp_l.append(path)  # Store the folder path
        csv_files_l.append(file)  # Store the filename
        csv_channel_l.append(ch)  # Store the channel name

# Create DataFrame directly from the collected data
df = pd.DataFrame({
    'fn': csv_files_l,
    'fp': csv_fp_l,
    'channel': csv_channel_l,
    'text': ['...'] * len(csv_files_l)  # Placeholder for text content
})

# Display the first few rows to verify
print(df.head())

# Save to CSV if needed
df.to_csv('//mwtx0006-nas01.bankofamerica.com/dcrs_acde$/POC/Gen_AI/ESO_Proj_May2025/Documents/texts/df_SCN.csv', index=False)
