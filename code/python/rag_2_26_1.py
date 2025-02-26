import json
import pandas as pd
from tqdm import tqdm
import time
import random
import ast
import httpx
import asyncio

# Initialize results list
classification_results = []

# Model configuration
model_name = "Meta-Llama-3.3-70B-Instruct"
model_params = {
    "temperature": 0.2,
    "top_p": 0.95,
    "logprobs": True,
    "prompt_type": "instruction"
}

# Configure retry settings
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 3
MAX_RETRY_DELAY = 60
BATCH_SIZE = 3  # Reduce to 3 to avoid overwhelming the server
PAUSE_DURATION = 5  # Increase to 5 seconds between batches

# Implementation of retry with exponential backoff specifically for your API
async def call_with_retry(func, *args, **kwargs):
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a server error
            if "internal server error" in error_str or "server" in error_str or "timeout" in error_str:
                wait_time = min(INITIAL_RETRY_DELAY * (2 ** retries) + random.uniform(0, 1), MAX_RETRY_DELAY)
                print(f"Server error encountered. Retrying in {wait_time:.2f} seconds... (Attempt {retries+1}/{MAX_RETRIES+1})")
                await asyncio.sleep(wait_time)
                retries += 1
                continue
            else:
                # If it's not a server error, raise it
                raise e
                
    raise Exception(f"Failed after {MAX_RETRIES} retries")

# Process documents in batches
async def process_documents():
    batch_count = 0
    successful_docs = 0
    failed_docs = 0
    
    for i, (idx, document) in enumerate(tqdm(testdf.iterrows(), total=len(testdf))):
        # Get document text
        text = document.get('truncated_text')
        
        # Skip if text is missing
        if not isinstance(text, str) or not text.strip():
            print(f"Skipping document {idx}: No valid text found")
            continue
        
        try:
            # Wrap the classification function with retry logic
            async def get_classification():
                return await combined_classification_rag(
                    user_query=text,
                    vector_db=vector_db,
                    model_name=model_name,
                    model_params=model_params,
                    top_k=3
                )
            
            # Call with retry
            output = await call_with_retry(get_classification)
            
            # Extract the result from the model response
            if isinstance(output, dict):
                # Use the direct path to the content based on your API structure
                if 'choices' in output and len(output['choices']) > 0:
                    if 'message' in output['choices'][0] and 'content' in output['choices'][0]['message']:
                        result = output['choices'][0]['message']['content'].strip()
                    else:
                        result = str(output['choices'][0]).strip()
                else:
                    result = str(output).strip()
            else:
                result = str(output).strip()
            
            # Parse classification result
            try:
                doc_parts = result.split(':') if ':' in result else [result, '']
                doc_type = doc_parts[0].strip()
                is_first_page = None
                
                if len(doc_parts) > 1:
                    first_page_text = doc_parts[1].strip().lower()
                    is_first_page = first_page_text == 'true' if first_page_text in ['true', 'false'] else None
            except Exception as parse_error:
                print(f"Error parsing result '{result}': {parse_error}")
                doc_type = "PARSE_ERROR"
                is_first_page = None
                
            # Add to results
            classification_results.append({
                'original_id': idx,
                'filepath': document.get('fp', ''),
                'filename': document.get('fn', ''),
                'actual_label': document.get('label', ''),
                'actual_first_page': document.get('first_pg', None),
                'predicted_label': doc_type,
                'predicted_first_page': is_first_page,
                'correct_label': document.get('label', '') == doc_type,
                'correct_first_page': document.get('first_pg', None) == is_first_page,
                'raw_result': result[:100]  # Store first 100 chars for debugging
            })
            
            successful_docs += 1
                
        except Exception as e:
            # Log errors but continue processing
            print(f"Error processing document {idx}: {str(e)}")
            classification_results.append({
                'original_id': idx,
                'filepath': document.get('fp', ''),
                'filename': document.get('fn', ''),
                'actual_label': document.get('label', ''),
                'actual_first_page': document.get('first_pg', None),
                'predicted_label': 'ERROR',
                'predicted_first_page': None,
                'correct_label': False,
                'correct_first_page': False,
                'error': str(e)
            })
            failed_docs += 1
        
        # Check if we've processed a batch
        batch_count += 1
        if batch_count >= BATCH_SIZE:
            # Create intermediate results DataFrame and save
            interim_df = pd.DataFrame(classification_results)
            interim_df.to_csv(f"document_classification_interim_{i}.csv", index=False)
            print(f"Saved interim results ({len(classification_results)} documents) to document_classification_interim_{i}.csv")
            print(f"Progress: {successful_docs} successful, {failed_docs} failed, {successful_docs + failed_docs}/{len(testdf)} total")
            
            # Pause between batches
            print(f"Pausing for {PAUSE_DURATION} seconds to avoid rate limiting...")
            await asyncio.sleep(PAUSE_DURATION)
            batch_count = 0

    # Final results
    results_df = pd.DataFrame(classification_results)
    
    # Calculate accuracy metrics for non-error results
    valid_results = results_df[results_df['predicted_label'] != 'ERROR']
    if len(valid_results) > 0:
        label_accuracy = valid_results['correct_label'].mean() * 100
        first_page_accuracy = valid_results['correct_first_page'].mean() * 100
        
        print(f"\nFinal Results:")
        print(f"Processed {len(valid_results)} out of {len(testdf)} documents successfully ({len(valid_results)/len(testdf)*100:.2f}%)")
        print(f"Document Type Classification Accuracy: {label_accuracy:.2f}%")
        print(f"First Page Classification Accuracy: {first_page_accuracy:.2f}%")
    else:
        print("No valid results to calculate accuracy")

    # Generate confusion matrix for document types
    if len(valid_results) > 0:
        print("\nConfusion Matrix for Document Types:")
        try:
            confusion = pd.crosstab(
                valid_results['actual_label'], 
                valid_results['predicted_label'], 
                rownames=['Actual'], 
                colnames=['Predicted']
            )
            print(confusion)
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")

    # Save final results to CSV
    results_df.to_csv("document_classification_results.csv", index=False)
    print("\nResults saved to document_classification_results.csv")

# Run the async function
# This needs to be executed in an async environment
# Example usage:
# await process_documents()
