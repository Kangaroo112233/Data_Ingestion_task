# Modified function to process the entire test dataset with better error handling
async def process_all_test_documents(testdf, vector_db, model_name="Meta-Llama-3.3-70B-Instruct"):
    # Create result columns
    results_df = testdf.copy()
    results_df["predicted_label"] = ""
    results_df["predicted_first_page"] = ""
    results_df["combined_result"] = ""
    results_df["is_correct_label"] = False
    results_df["is_correct_first_page"] = False
    results_df["error"] = ""
    
    # Set model parameters
    model_params = {
        "temperature": 0.2,
        "top_p": 0.95
    }
    
    # Process each document in batches to avoid overwhelming the server
    print(f"Processing {len(testdf)} documents...")
    
    # Define batch size (adjust based on server capacity)
    batch_size = 5
    successful = 0
    failed = 0
    
    for start_idx in range(0, len(testdf), batch_size):
        end_idx = min(start_idx + batch_size, len(testdf))
        print(f"Processing batch {start_idx//batch_size + 1}: documents {start_idx}-{end_idx-1}")
        
        for idx in range(start_idx, end_idx):
            row = testdf.iloc[idx]
            doc_text = row["truncated_text"]
            true_label = row["label"]
            true_first_page = row["first_pg"]
            
            try:
                # Get combined classification
                combined_result = await combined_classification_rag(
                    user_query=doc_text,
                    vector_db=vector_db,
                    model_name=model_name,
                    model_params=model_params,
                    top_k=3
                )
                
                # Handle different response formats
                try:
                    # Parse combined result - could be in different formats
                    if isinstance(combined_result, dict):
                        # If we got a dict response, extract the content
                        if 'choices' in combined_result and len(combined_result['choices']) > 0:
                            content = combined_result['choices'][0]['message']['content']
                        else:
                            content = str(combined_result)
                        combined_result = content
                    
                    # Try to parse the content in format "DocumentType:FirstPageStatus"
                    if ':' in combined_result:
                        parts = combined_result.split(":", 1)
                        predicted_label = parts[0].strip()
                        predicted_first_page = parts[1].strip().lower() == "true"
                    else:
                        # Fallback: try to extract document type directly
                        if any(doc_type.lower() in combined_result.lower() for doc_type in 
                               ["bank statement", "paystub", "w2", "other"]):
                            for doc_type in ["bank statement", "paystub", "w2", "other"]:
                                if doc_type.lower() in combined_result.lower():
                                    predicted_label = doc_type.capitalize()
                                    break
                        else:
                            predicted_label = "Unknown"
                        
                        # Try to find first page status
                        if "true" in combined_result.lower():
                            predicted_first_page = True
                        elif "false" in combined_result.lower():
                            predicted_first_page = False
                        else:
                            predicted_first_page = None
                    
                    # Store results
                    results_df.at[idx, "predicted_label"] = predicted_label
                    results_df.at[idx, "predicted_first_page"] = predicted_first_page
                    results_df.at[idx, "combined_result"] = combined_result
                    
                    # Check if predictions are correct
                    if predicted_label and predicted_label.lower() == true_label.lower():
                        results_df.at[idx, "is_correct_label"] = True
                    
                    if predicted_first_page is not None and predicted_first_page == true_first_page:
                        results_df.at[idx, "is_correct_first_page"] = True
                    
                    successful += 1
                    
                except Exception as parsing_error:
                    results_df.at[idx, "error"] = f"Parsing error: {str(parsing_error)}"
                    results_df.at[idx, "combined_result"] = str(combined_result)
                    failed += 1
                    
            except Exception as e:
                results_df.at[idx, "error"] = f"API error: {str(e)}"
                failed += 1
            
            # Add a small delay between requests to avoid overwhelming the server
            await asyncio.sleep(0.5)
    
    print(f"\nProcessing complete: {successful} successful, {failed} failed")
    return results_df

# Function to calculate metrics on partial results
def calculate_metrics(results_df):
    """Calculate metrics on available results, ignoring errors"""
    # Filter out rows with errors
    valid_results = results_df[results_df["error"] == ""]
    
    if len(valid_results) == 0:
        print("No valid results to calculate metrics")
        return
    
    # Calculate metrics
    label_accuracy = valid_results["is_correct_label"].mean() * 100
    first_page_accuracy = valid_results["is_correct_first_page"].mean() * 100
    
    # Count how many rows had both correct
    both_correct = (valid_results["is_correct_label"] & valid_results["is_correct_first_page"]).sum()
    overall_accuracy = (both_correct / len(valid_results)) * 100
    
    print(f"\nResults based on {len(valid_results)} valid documents:")
    print(f"Document Type Accuracy: {label_accuracy:.2f}%")
    print(f"First Page Detection Accuracy: {first_page_accuracy:.2f}%")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    
    # Breakdown by document type
    print("\nAccuracy by Document Type:")
    for doc_type in valid_results["label"].unique():
        type_mask = valid_results["label"] == doc_type
        if type_mask.sum() > 0:
            type_accuracy = valid_results.loc[type_mask, "is_correct_label"].mean() * 100
            print(f"{doc_type}: {type_accuracy:.2f}% ({valid_results.loc[type_mask, 'is_correct_label'].sum()} / {type_mask.sum()})")

# Run the processing function with error handling
async def main():
    try:
        print("Starting document classification...")
        
        # Process a smaller subset for testing if needed
        # sample_size = 50
        # sample_testdf = testdf.sample(sample_size, random_state=42)
        # results_df = await process_all_test_documents(sample_testdf, vector_db)
        
        # Process all documents
        results_df = await process_all_test_documents(testdf, vector_db)
        
        # Calculate metrics on whatever results we have
        calculate_metrics(results_df)
        
        # Save results to CSV
        results_df.to_csv("classification_results.csv", index=False)
        print("Results saved to classification_results.csv")
        
        return results_df
        
    except Exception as e:
        print(f"Error in main processing: {str(e)}")
        import traceback
        traceback.print_exc()

# Execute the main function
import asyncio
resultdf = asyncio.run(main())



CLASSIFIER_SYSTEM_PROMPT = """
You are a document classification assistant. Given the retrieved document content, determine both:
1) The document type (Bank Statement, Paystub, W2, or Other).
2) Whether this is the first page of the document (True/False).

Use only the retrieved text to make your decision.

IMPORTANT: You MUST respond ONLY with a JSON object in the following exact format, with no additional text before or after:
{
  "document_type": "Bank Statement|Paystub|W2|Other",
  "is_first_page": true|false
}

### Retrieved Context ###
{retrieved_chunks}

### User Query ###
{user_query}
"""

async def combined_classification_rag(
    user_query: str,
    vector_db,
    model_name: str,
    model_params: dict,
    top_k: int = 3
) -> str:
    """
    1) Retrieves context from Chroma DB.
    2) Constructs final prompt with CLASSIFIER_SYSTEM_PROMPT + context.
    3) Calls the remote model for a combined classification result with JSON output.
    """
    # Step 1: Retrieve relevant chunks
    context = rag_retrieve(user_query, vector_db, k=top_k)
    
    # Step 2: Build the user portion of the prompt
    final_user_query = f"Document Text:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    
    # Updated system prompt that enforces JSON output format
    JSON_CLASSIFIER_SYSTEM_PROMPT = """
    You are a document classification assistant. Given the retrieved document content, determine both:
    1) The document type (Bank Statement, Paystub, W2, or Other).
    2) Whether this is the first page of the document (True/False).

    Use only the retrieved text to make your decision.

    IMPORTANT: You MUST respond ONLY with a JSON object in the following exact format, with no additional text before or after:
    {
      "document_type": "Bank Statement|Paystub|W2|Other",
      "is_first_page": true|false
    }
    """
    
    # Step 3: Call the model service API
    # Force JSON output format using model parameters
    enhanced_params = model_params.copy()
    enhanced_params["response_format"] = {"type": "json_object"}
    
    result = await prompt_model(
        query=final_user_query,
        system_prompt=JSON_CLASSIFIER_SYSTEM_PROMPT,
        model_name=model_name,
        model_params=enhanced_params,
        top_k=top_k
    )
    
    # Extract the JSON content from the response
    try:
        if isinstance(result, dict) and 'choices' in result:
            content = result['choices'][0]['message']['content']
            
            # Parse the content to ensure it's valid JSON
            import json
            parsed_json = json.loads(content)
            
            # Create a clean, minimal JSON with only the required fields
            clean_json = {
                "document_type": parsed_json.get("document_type", "Unknown"),
                "is_first_page": parsed_json.get("is_first_page", False)
            }
            
            # Return the clean JSON as a string
            return json.dumps(clean_json)
        else:
            # Fallback for unexpected responses
            return json.dumps({
                "document_type": "Unknown",
                "is_first_page": False
            })
    except Exception as e:
        # Handle any parsing errors
        print(f"Error parsing model response: {e}")
        return json.dumps({
            "document_type": "Error",
            "is_first_page": False
        })
