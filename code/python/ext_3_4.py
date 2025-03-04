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
