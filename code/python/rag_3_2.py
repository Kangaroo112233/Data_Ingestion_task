async def apply_combined_classification_to_dataset(testdf, vector_db, model_name, batch_size=10):
    """
    Apply combined classification to each record in the test dataset
    
    Args:
        testdf: Test dataframe
        vector_db: Vector database for retrieval
        model_name: Name of the model to use
        batch_size: Number of records to process at once (to avoid rate limits)
        
    Returns:
        DataFrame with added prediction columns
    """
    # Create a copy of the dataframe to avoid modifying the original
    resultdf = testdf.copy()
    
    # Add new columns for predictions
    resultdf['predicted_label'] = None
    resultdf['predicted_first_page'] = None
    
    # Set model parameters
    model_params = {
        "temperature": 0.2,
        "top_p": 0.95,
        "logprobs": False,  # Set to True if you want log probabilities
    }
    
    # Process in smaller batches to avoid potential rate limits
    total_records = len(resultdf)
    print(f"Processing {total_records} records...")
    
    # Create a progress tracker
    from tqdm.notebook import tqdm
    
    for i in tqdm(range(0, total_records, batch_size)):
        batch_end = min(i + batch_size, total_records)
        batch = resultdf.iloc[i:batch_end]
        
        for idx, row in batch.iterrows():
            try:
                # Get the truncated text for the record
                document_text = row["truncated_text"]
                
                # Apply combined classification
                result = await combined_classification_rag(
                    user_query=document_text,
                    vector_db=vector_db,
                    model_name=model_name,
                    model_params=model_params,
                    top_k=3
                )
                
                # Parse the result to extract document type and first page status
                # Assuming format is "Document Type:True/False"
                content = result['choices'][0]['message']['content']
                
                # Extract document type and first page status
                # Handle different response formats
                if ':' in content:
                    # Format: "Document Type:True/False"
                    parts = content.strip().split(':')
                    doc_type = parts[0].strip()
                    first_page = parts[1].strip().lower() == 'true'
                else:
                    # Format might be different, try to parse as best as possible
                    if 'bank statement' in content.lower():
                        doc_type = 'Bank Statement'
                    elif 'paystub' in content.lower():
                        doc_type = 'Paystub'
                    elif 'w2' in content.lower():
                        doc_type = 'W2'
                    else:
                        doc_type = 'Other'
                    
                    first_page = 'true' in content.lower() and not 'false' in content.lower()
                
                # Store the predictions
                resultdf.at[idx, 'predicted_label'] = doc_type
                resultdf.at[idx, 'predicted_first_page'] = first_page
                
                # Optional: Add a small delay to avoid rate limits if needed
                # await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing record {idx}: {e}")
                # Continue with the next record
    
    # Calculate accuracy metrics
    label_accuracy = (resultdf['label'] == resultdf['predicted_label']).mean()
    first_page_accuracy = (resultdf['first_pg'].astype(str).str.lower() == resultdf['predicted_first_page'].astype(str).str.lower()).mean()
    
    print(f"Label Prediction Accuracy: {label_accuracy:.4f}")
    print(f"First Page Prediction Accuracy: {first_page_accuracy:.4f}")
    
    # Show confusion matrix for labels
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create confusion matrix for document types
    print("\nDocument Type Confusion Matrix:")
    label_cm = confusion_matrix(resultdf['label'], resultdf['predicted_label'], labels=sorted(resultdf['label'].unique()))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(label_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(resultdf['label'].unique()),
                yticklabels=sorted(resultdf['label'].unique()))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Document Type Confusion Matrix')
    plt.show()
    
    # Create confusion matrix for first page detection
    print("\nFirst Page Detection Confusion Matrix:")
    first_page_cm = confusion_matrix(resultdf['first_pg'].astype(str), 
                                    resultdf['predicted_first_page'].astype(str),
                                    labels=['True', 'False'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(first_page_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['True', 'False'],
                yticklabels=['True', 'False'])
    plt.ylabel('True First Page')
    plt.xlabel('Predicted First Page')
    plt.title('First Page Detection Confusion Matrix')
    plt.show()
    
    return resultdf

# Run the classification on the test dataset
# You might want to use a smaller subset for testing first
test_subset = testdf.sample(50, random_state=42)  # Start with a small sample of 50 records

# Run the classification
resultdf = await apply_combined_classification_to_dataset(
    test_subset,  # You can replace this with testdf for the full dataset
    vector_db=vector_db,
    model_name="Meta-Llama-3.3-70B-Instruct",
    batch_size=5  # Small batch size to avoid potential rate limits
)

# Display the results
print("\nSample of classification results:")
print(resultdf[['label', 'predicted_label', 'first_pg', 'predicted_first_page']].head(10))

# Save the results to CSV if needed
# resultdf.to_csv("classification_results.csv", index=False)

# For the full test set, you can run:
# resultdf_full = await apply_combined_classification_to_dataset(testdf, vector_db, "Meta-Llama-3.3-70B-Instruct", batch_size=10)
