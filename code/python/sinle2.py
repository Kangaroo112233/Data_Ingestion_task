import os
import pandas as pd

# Initialize empty lists
csv_files_l = list()
csv_fp_l = list()
csv_channel_l = list()
text_l = list()

# Base directory path
docdir = "//mwtx0006-nas01.bankofamerica.com/dcrs_acde$/POC/Gen_AI/ESO_Proj_May2025/Documents/texts/"
print(f"Base directory: {docdir}")

# Process all channels
channels = ["EML", "FAX", "SCN", "WIN"]
for ch in channels:
    channel_path = os.path.join(docdir, ch)
    print(f"Processing channel: {ch} at {channel_path}")
    
    try:
        # Get all CSV files in the channel folder
        files = [f for f in os.listdir(channel_path) if f.endswith('.csv')]
        print(f"Found {len(files)} CSV files in {ch}")
        
        # Process each file
        for file in files:
            file_path = os.path.join(channel_path, file)
            
            # Read the CSV file to extract text content
            try:
                # Attempt to read the first few lines to get text content
                csv_data = pd.read_csv(file_path, nrows=1)
                if 'text' in csv_data.columns:
                    text = csv_data['text'].iloc[0]
                else:
                    # If no 'text' column, use first column's first value
                    text = str(csv_data.iloc[0, 0])
            except Exception as e:
                print(f"Warning: Could not read text from {file}: {e}")
                text = "..."  # Placeholder if reading fails
            
            csv_files_l.append(file)
            csv_fp_l.append(channel_path)
            csv_channel_l.append(ch)
            text_l.append(text)
            
    except Exception as e:
        print(f"Error processing channel {ch}: {e}")

# Create DataFrame
df = pd.DataFrame({
    'fn': csv_files_l,
    'fp': csv_fp_l,
    'channel': csv_channel_l,
    'text': text_l
})

# Display info
print(f"Created DataFrame with {len(df)} rows")
print(df.head())

# Save to Vector db/RAG_chroma directory
output_dir = "Vector db/RAG_chroma/"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
output_path = os.path.join(output_dir, "df_all_channels.csv")
df.to_csv(output_path, index=False)
print(f"Saved DataFrame to {output_path}")

# Also create a variable to easily load this file later
dataset = output_path
print(f"To load this file later, use: pd.read_csv('{dataset}')")
