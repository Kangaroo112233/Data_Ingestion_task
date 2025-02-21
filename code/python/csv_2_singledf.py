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

# For each channel
channels = ["EML", "FAX", "SCN", "WIN"]
for ch in channels:
    # Path to the channel
    path = os.path.join(docdir, ch)
    print(f"Processing channel: {ch}")
    
    try:
        # Get all PDF files directly in the channel folder
        pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
        
        # Process each file
        for file in pdf_files:
            file_path = os.path.join(path, file)
            csv_fp_l.append(path)  # Store the folder path
            csv_files_l.append(file)  # Store the filename
            csv_channel_l.append(ch)  # Store the channel name
            
            # Placeholder for text - you might want to extract actual text from PDFs
            # This would require a PDF text extraction library
            data.append("...")
    except Exception as e:
        print(f"Error processing channel {ch}: {e}")

# Create DataFrame directly from the collected data
df = pd.DataFrame({
    'fn': csv_files_l,
    'fp': csv_fp_l,
    'channel': csv_channel_l,
    'text': data
})

# Display the first few rows to verify
print(df.head())

# Save to CSV if needed
output_path = os.path.join(docdir, "df_all_channels.csv")
df.to_csv(output_path, index=False)
print(f"Saved DataFrame to {output_path}")
