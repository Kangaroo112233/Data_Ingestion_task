import os
import pandas as pd

# Define base directory
out_root = r"//nwtx0006-nas01.bankofamerica.com/dcrs_acde$/POC/Gen_AI/ESO_Proj_May2025/Documents/UAT_groundtruth_DIIVO"
doc_folder = os.path.join(out_root, "business_personal_signature_card")

# Initialize data collections
data = {
    "fn": [],
    "fp": [],
    "folder": [],
    "text": []
}

# Use proper path joining
pdf_dir = os.path.join(doc_folder, "pdfs")
text_dir = os.path.join(doc_folder, "texts")

# Check if directory exists before proceeding
if os.path.exists(pdf_dir):
    try:
        # List PDF files
        pdfs = os.listdir(pdf_dir)
        print(f"Found {len(pdfs)} PDF files")
        
        # Process each PDF file
        for pdf in pdfs:
            # Add to data collection
            data["fn"].append(pdf)
            data["fp"].append(os.path.join(pdf_dir, pdf))
            data["folder"].append(os.path.basename(pdf_dir))
            
            # Corresponding text file
            text_file = pdf + ".pdf.csv"
            text_path = os.path.join(text_dir, text_file)
            
            if os.path.exists(text_path):
                try:
                    # OCR Text
                    tdf = pd.read_csv(text_path)
                    tdf = tdf.fillna('')
                    text_pages_list = [tdf.columns.tolist()] + tdf.values.tolist()
                    
                    full_text = ""
                    for page in text_pages_list:
                        full_text += page[0] + "\n"
                    data["text"].append(full_text)
                except Exception as e:
                    print(f"Error reading {text_path}: {e}")
                    data["text"].append("")
            else:
                data["text"].append("")
    except Exception as e:
        print(f"Error processing files: {e}")
else:
    print(f"Directory not found: {pdf_dir}")

# Create DataFrame
df = pd.DataFrame(data)

# Save results
output_file = os.path.join(doc_folder, "business_personal_signature_card_100_2.csv")
df.to_csv(output_file, index=False)
print(f"Saved results to {output_file}")
