async def apply_combined_classification_to_dataset(testdf, vector_db, model_name, batch_size=5):
    """
    Apply combined classification to each record in the test dataset with improved error handling
    
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
    resultdf['error'] = None  # Add column to track errors
    
    # Set model parameters
    model_params = {
        "temperature": 0.2,
        "top_p": 0.95,
    }
    
    # Process in smaller batches to avoid potential rate limits
    total_records = len(resultdf)
    print(f"Processing {total_records} records...")
    
    # Create a progress tracker
    from tqdm.notebook import tqdm
    import asyncio
    
    # Count successful and failed classifications
    success_count = 0
    error_count = 0
    
    for i in tqdm(range(0, total_records, batch_size)):
        batch_end = min(i + batch_size, total_records)
        batch = resultdf.iloc[i:batch_end]
        
        for idx, row in batch.iterrows():
            try:
                # Get the truncated text for the record
                document_text = row["truncated_text"]
                
                # Apply direct classification without using the combined_classification_rag function
                # This helps us have more control over error handling
                
                # Step 1: Retrieve context
                context = rag_retrieve(document_text, vector_db, k=3)
                
                # Step 2: Build the prompt
                final_user_query = f"Document Text:\n{context}\n\nQuestion: {document_text}\nAnswer:"
                
                # Step 3: Call the model API directly
                try:
                    # Create the request payload
                    request_body = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                            {"role": "user", "content": final_user_query}
                        ],
                        "temperature": model_params.get("temperature", 0.2),
                        "top_p": model_params.get("top_p", 0.95)
                    }
                    
                    # Make direct API call with proper error handling
                    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                        response = await client.post(
                            phoenix_genai_service_url, 
                            headers=headers, 
                            json=request_body
                        )
                        
                        # Check response status
                        if response.status_code != 200:
                            raise Exception(f"API error: {response.status_code} {response.text}")
                            
                        # Parse response
                        response_data = response.json()
                        
                        # Extract the content
                        if 'choices' in response_data and len(response_data['choices']) > 0:
                            content = response_data['choices'][0]['message']['content'].strip()
                        else:
                            raise Exception("No content in response")
                            
                        # Parse the classification result
                        doc_type = "Unknown"
                        is_first_page = False
                        
                        # Try to parse the result (Document Type:First Page Status)
                        if ':' in content:
                            parts = content.split(':')
                            doc_type = parts[0].strip()
                            is_first_page_str = parts[1].strip().lower()
                            is_first_page = is_first_page_str == 'true'
                        else:
                            # Fallback parsing logic
                            lower_content = content.lower()
                            if 'bank statement' in lower_content:
                                doc_type = 'Bank Statement'
                            elif 'paystub' in lower_content:
                                doc_type = 'Paystub'
                            elif 'w2' in lower_content:
                                doc_type = 'W2'
                            else:
                                doc_type = 'Other'
                            
                            is_first_page = 'true' in lower_content and not 'false' in lower_content
                        
                        # Store results
                        resultdf.at[idx, 'predicted_label'] = doc_type
                        resultdf.at[idx, 'predicted_first_page'] = is_first_page
                        resultdf.at[idx, 'error'] = None
                        success_count += 1
                        
                        # Add a small delay to avoid overloading the server
                        await asyncio.sleep(0.2)
                        
                except Exception as api_error:
                    # Handle API-specific errors
                    error_msg = f"API error: {str(api_error)}"
                    resultdf.at[idx, 'error'] = error_msg
                    print(f"Error with record {idx}: {error_msg}")
                    error_count += 1
                    await asyncio.sleep(0.5)  # Longer delay after an error
                    
            except Exception as e:
                # Handle general errors
                error_msg = f"General error: {str(e)}"
                resultdf.at[idx, 'error'] = error_msg
                print(f"Error processing record {idx}: {error_msg}")
                error_count += 1
                await asyncio.sleep(0.5)  # Longer delay after an error
    
    # Print summary
    print(f"\nClassification complete!")
    print(f"Successfully classified: {success_count} records")
    print(f"Errors encountered: {error_count} records")
    
    # Calculate accuracy metrics only for successfully classified records
    successful_df = resultdf[resultdf['error'].isna()]
    if len(successful_df) > 0:
        label_accuracy = (successful_df['label'] == successful_df['predicted_label']).mean()
        first_page_accuracy = (successful_df['first_pg'].astype(str).str.lower() == 
                              successful_df['predicted_first_page'].astype(str).str.lower()).mean()
        
        print(f"Label Prediction Accuracy: {label_accuracy:.4f}")
        print(f"First Page Prediction Accuracy: {first_page_accuracy:.4f}")
    else:
        print("No successful classifications to calculate accuracy")
    
    return resultdf

# Run the classification on a smaller subset first to test
test_subset = testdf.sample(10, random_state=42)  # Start with just 10 records

# Run the classification with improved error handling
import asyncio
resultdf = await apply_combined_classification_to_dataset(
    test_subset,  # You can replace this with testdf for the full dataset later
    vector_db=vector_db,
    model_name="Meta-Llama-3.3-70B-Instruct",
    batch_size=2  # Small batch size to avoid overwhelming the API
)

# Display the results
print("\nSample of classification results:")
print(resultdf[['label', 'predicted_label', 'first_pg', 'predicted_first_page', 'error']].head(10))

# If you want to save the results:
# resultdf.to_csv("classification_results.csv", index=False)

# Once the small batch works, you can try the full dataset or a larger sample:
# full_resultdf = await apply_combined_classification_to_dataset(
#     testdf,
#     vector_db=vector_db,
#     model_name="Meta-Llama-3.3-70B-Instruct",
#     batch_size=10
# )
