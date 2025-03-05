# Assuming df already has the 'extracted_references' column

# 1. Determine the expected reference column
# From Image 2, it looks like the expected references should be in 'reference_number' column
# But the actual format might need cleaning to match your extracted format

# 2. Create a validation column
df['is_match'] = df.apply(lambda row: 
    # If extracted_references is 'None', check if reference_number is empty/NaN
    row['extracted_references'] == row['reference_number'] 
    if row['extracted_references'] != 'None' 
    else pd.isna(row['reference_number']) or row['reference_number'] == '' or row['reference_number'] == 'NaN',
    axis=1
)

# 3. For more detailed validation (optional)
# Create a column to indicate the type of match/mismatch
df['validation_status'] = df.apply(lambda row:
    'MATCH' if row['is_match'] else
    'MISSING' if row['extracted_references'] == 'None' and not pd.isna(row['reference_number']) else
    'FALSE_POSITIVE' if row['extracted_references'] != 'None' and pd.isna(row['reference_number']) else
    'MISMATCH',
    axis=1
)

# 4. Calculate validation statistics
total_records = len(df)
matches = df['is_match'].sum()
match_percentage = (matches / total_records) * 100

print(f"Total records: {total_records}")
print(f"Matching records: {matches} ({match_percentage:.2f}%)")

# 5. To save the validation results
df.to_excel('extraction_validation_results.xlsx', index=False)


# For more specific validation of date-pattern reference IDs
def validate_reference_format(extracted, expected):
    # Basic equality check
    if extracted == expected:
        return True
    
    # If extracted is 'None' but expected is not empty
    if extracted == 'None' and expected and expected != 'NaN':
        return False
    
    # If extracted found something but expected is empty
    if extracted != 'None' and (not expected or expected == 'NaN'):
        return False
    
    # Special case: Check if the date portion matches even if other parts differ
    # Example: Extract date portion (YYYYMMDD) from both and compare
    if extracted != 'None' and expected:
        # Extract date pattern from strings
        import re
        date_pattern = r'(\d{8})'  # Matches 8 consecutive digits (YYYYMMDD)
        
        extracted_date = re.search(date_pattern, extracted)
        expected_date = re.search(date_pattern, expected)
        
        if extracted_date and expected_date:
            return extracted_date.group(0) == expected_date.group(0)
    
    return False

# Apply this more detailed validation
df['is_format_match'] = df.apply(lambda row: 
    validate_reference_format(row['extracted_references'], row['reference_number']),
    axis=1
)


CLASSIFIER_SYSTEM_PROMPT = """
You are a document classification assistant. Given the retrieved document content, determine both:
1) The document type (Bank Statement, Paystub, W2, or Other).
2) Whether this is the first page of the document (true/false).

Use only the retrieved text to make your decision.

### Retrieved Context ###
{retrieved_chunks}

### User Query ###
{user_query}

### Answer Format ###
Return your answer as a valid JSON object with two keys: "document_label" and "is_first_page". 
For example: {"document_label": "Bank Statement", "is_first_page": true}
"""
import pandas as pd
import asyncio
import json

async def classify_document(row, vector_db, model_name, model_params):
    """
    Processes a single row by calling the combined classification model
    and extracting the document label and first-page classification.
    """
    try:
        result = await combined_classification_rag(
            user_query=row["truncated_text"],
            vector_db=vector_db,
            model_name=model_name,
            model_params=model_params,
            top_k=3
        )

        # Extract the response JSON from the model
        raw_text = result["choices"][0]["message"]["content"]
        classification_dict = json.loads(raw_text)

        return classification_dict["document_label"], classification_dict["is_first_page"]

    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return None, None  # Return None values in case of error

async def process_dataframe(testdf, vector_db, model_name, model_params):
    """
    Applies document classification to the entire testdf asynchronously
    and returns the updated DataFrame with new columns.
    """
    tasks = [
        classify_document(row, vector_db, model_name, model_params)
        for _, row in testdf.iterrows()
    ]
    
    results = await asyncio.gather(*tasks)

    # Store results back into the DataFrame
    testdf["document_label"], testdf["is_first_page"] = zip(*results)

    return testdf

# Define model parameters
model_name = "Meta-Llama-3.3-70B-Instruct"
model_params = {
    "temperature": 0.2,
    "top_p": 0.95
}

# Run the classification over the entire DataFrame
resultdf = asyncio.run(process_dataframe(testdf, vector_db, model_name, model_params))

# Display updated DataFrame
import ace_tools as tools
tools.display_dataframe_to_user(name="Classified Documents", dataframe=resultdf)
