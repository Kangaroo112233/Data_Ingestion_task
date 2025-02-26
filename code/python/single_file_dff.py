import os
import pandas as pd

# Define base directory
out_root = r"\\nwtx0006-nas01.bankofamerica.com\dcrs_acde$\POC\Gen_AI\ESO_Proj_May2025\Documents\UAT_groundtruth_DIIVO"
doc_folder = os.path.join(out_root, "Death-Certificate")

# Initialize data collections
data = {
    "fn": [],
    "fp": [],
    "channel": [],
    "folder": [],
    "text": []
}

# Process each channel
channels = ["EML", "FAX", "SCN", "WIN"]
for channel in channels:
    # Use proper path joining
    pdf_dir = os.path.join(doc_folder, "pdfs", channel)
    text_dir = os.path.join(doc_folder, "texts", channel)
    
    # Check if directory exists before proceeding
    if os.path.exists(pdf_dir):
        try:
            # List PDF files
            pdfs = os.listdir(pdf_dir)
            print(f"Found {len(pdfs)} PDF files in {channel}")
            
            # Process each PDF file
            for pdf in pdfs:
                # Add to data collection
                data["fn"].append(pdf)
                data["fp"].append(os.path.join(pdf_dir, pdf))
                data["channel"].append(channel)
                data["folder"].append(os.path.basename(pdf_dir))
                
                # Corresponding text file
                text_file = pdf + ".pdf.csv"
                text_path = os.path.join(text_dir, text_file)
                
                if os.path.exists(text_path):
                    try:
                        # Read text content
                        tdf = pd.read_csv(text_path)
                        full_text = "\n".join(tdf.values.flatten())
                        data["text"].append(full_text)
                    except Exception as e:
                        print(f"Error reading {text_path}: {e}")
                        data["text"].append("")
                else:
                    data["text"].append("")
        except Exception as e:
            print(f"Error processing {channel}: {e}")
    else:
        print(f"Directory not found: {pdf_dir}")

# Create DataFrame
df = pd.DataFrame(data)

# Save results
output_file = os.path.join(doc_folder, "Death-Certificate_200.csv")
df.to_csv(output_file, index=False)
print(f"Saved results to {output_file}")
