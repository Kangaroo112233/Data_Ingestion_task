import json
import pandas as pd
from tqdm import tqdm  # For progress tracking

# Initialize results list
classification_results = []

# Set model parameters
model_name = "Meta-Llama-3.3-70B-Instruct"
model_params = {
    "temperature": 0.2,
    "top_p": 0.95,
    "logprobs": True,
    "prompt_type": "instruction"
}

# Process each document in the test set with a progress bar
for idx, document in tqdm(testdf.iterrows(), total=len(testdf)):
    # Get document text - using truncated_text to match your workflow
    text = document['truncated_text']
    
    # Call the combined classification (both document type and first-page)
    try:
        # Using combined_classification_rag function as shown in your previous code
        output = await combined_classification_rag(
            user_query=text,
            vector_db=vector_db,
            model_name=model_name,
            model_params=model_params,
            top_k=3
        )
        
        # Extract the result (format should be "DocType:FirstPage")
        result = output
        
        # Parse the result
        doc_parts = result.split(':')
        if len(doc_parts) == 2:
            doc_type = doc_parts[0].strip()
            is_first_page = doc_parts[1].strip().lower() == 'true'
        else:
            doc_type = result
            is_first_page = None
            
        # Add to results with original document metadata
        classification_results.append({
            'original_id': idx,
            'filepath': document.get('fp', ''),
            'filename': document.get('fn', ''),
            'actual_label': document['label'],  # Ground truth label
            'actual_first_page': document['first_pg'],  # Ground truth first page
            'predicted_label': doc_type,
            'predicted_first_page': is_first_page,
            'correct_label': document['label'] == doc_type,
            'correct_first_page': document['first_pg'] == is_first_page
        })
            
    except Exception as e:
        # Log errors but continue processing
        print(f"Error processing document {idx}: {e}")
        classification_results.append({
            'original_id': idx,
            'filepath': document.get('fp', ''),
            'filename': document.get('fn', ''),
            'actual_label': document['label'],
            'actual_first_page': document['first_pg'],
            'predicted_label': 'ERROR',
            'predicted_first_page': None,
            'correct_label': False,
            'correct_first_page': False,
            'error': str(e)
        })

# Convert results to DataFrame
results_df = pd.DataFrame(classification_results)

# Calculate accuracy metrics
label_accuracy = results_df['correct_label'].mean() * 100
first_page_accuracy = results_df['correct_first_page'].mean() * 100

print(f"Document Type Classification Accuracy: {label_accuracy:.2f}%")
print(f"First Page Classification Accuracy: {first_page_accuracy:.2f}%")

# Generate confusion matrix for document types
print("\nConfusion Matrix for Document Types:")
confusion = pd.crosstab(
    results_df['actual_label'], 
    results_df['predicted_label'], 
    rownames=['Actual'], 
    colnames=['Predicted']
)
print(confusion)

# Save results to CSV
results_df.to_csv("document_classification_results.csv", index=False)
print("\nResults saved to document_classification_results.csv")




import json
import pandas as pd
from tqdm import tqdm

# Initialize results list
classification_results = []

# Set model parameters
model_name = "Meta-Llama-3.3-70B-Instruct"
model_params = {
    "temperature": 0.2,
    "top_p": 0.95,
    "logprobs": True,
    "prompt_type": "instruction"
}

# Process each document in the test set with a progress bar
for idx, document in tqdm(testdf.iterrows(), total=len(testdf)):
    # Get document text - using truncated_text to match your workflow
    text = document.get('truncated_text')
    
    # Skip if text is missing
    if not isinstance(text, str) or not text.strip():
        print(f"Skipping document {idx}: No valid text found")
        continue
    
    # Call the combined classification (both document type and first-page)
    try:
        # Using combined_classification_rag function
        output = await combined_classification_rag(
            user_query=text,
            vector_db=vector_db,
            model_name=model_name,
            model_params=model_params,
            top_k=3
        )
        
        # Check response type and format
        if isinstance(output, dict):
            # If output is a dictionary (likely API response)
            if 'choices' in output and len(output['choices']) > 0:
                # Extract from the choices array
                if isinstance(output['choices'][0], dict) and 'message' in output['choices'][0]:
                    content = output['choices'][0]['message'].get('content', '')
                    result = content.strip()
                else:
                    result = str(output['choices'][0]).strip()
            else:
                # Fallback if choices not found
                result = str(output).strip()
        else:
            # If output is already a string
            result = str(output).strip()
        
        # Parse the result (looking for format like "DocType:True/False")
        doc_parts = result.split(':') if ':' in result else [result, '']
        doc_type = doc_parts[0].strip()
        is_first_page = None
        
        if len(doc_parts) > 1:
            first_page_text = doc_parts[1].strip().lower()
            is_first_page = first_page_text == 'true' if first_page_text in ['true', 'false'] else None
            
        # Add to results with original document metadata
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
            'raw_result': result[:100]  # Store first 100 chars of raw result for debugging
        })
            
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

# Convert results to DataFrame
results_df = pd.DataFrame(classification_results)

# Calculate accuracy metrics for non-error results
valid_results = results_df[results_df['predicted_label'] != 'ERROR']
if len(valid_results) > 0:
    label_accuracy = valid_results['correct_label'].mean() * 100
    first_page_accuracy = valid_results['correct_first_page'].mean() * 100
    
    print(f"Processed {len(valid_results)} out of {len(testdf)} documents successfully")
    print(f"Document Type Classification Accuracy: {label_accuracy:.2f}%")
    print(f"First Page Classification Accuracy: {first_page_accuracy:.2f}%")
else:
    print("No valid results to calculate accuracy")

# Generate confusion matrix for document types (only for valid predictions)
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

# Save results to CSV
results_df.to_csv("document_classification_results.csv", index=False)
print("\nResults saved to document_classification_results.csv")


import json
import pandas as pd
from tqdm import tqdm
import time
import random

# Initialize results list
classification_results = []

# Set model parameters
model_name = "Meta-Llama-3.3-70B-Instruct"
model_params = {
    "temperature": 0.2,
    "top_p": 0.95,
    "logprobs": True,
    "prompt_type": "instruction"
}

# Helper function to implement exponential backoff
def call_with_retry(func, *args, max_retries=5, **kwargs):
    retries = 0
    while retries <= max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait_time = (2 ** retries) + random.uniform(0, 1)  # Exponential backoff with jitter
            
            # Check if it's a server error (could be more specific based on your error format)
            if "Internal Server Error" in str(e) or "Server" in str(e):
                print(f"Server error encountered. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                retries += 1
                continue
            else:
                # If it's not a server error, raise it immediately
                raise e
                
    # If we've exhausted all retries
    raise Exception(f"Failed after {max_retries} retries")

# Batch size - process this many documents at once, then pause
BATCH_SIZE = 5
# Pause duration between batches (seconds)
PAUSE_DURATION = 3

# Process documents in batches
batch_count = 0
for i, (idx, document) in enumerate(tqdm(testdf.iterrows(), total=len(testdf))):
    # Get document text
    text = document.get('truncated_text')
    
    # Skip if text is missing
    if not isinstance(text, str) or not text.strip():
        print(f"Skipping document {idx}: No valid text found")
        continue
    
    # Call the combined classification with retry logic
    try:
        # Use the retry wrapper around your classification function
        async def perform_classification():
            return await combined_classification_rag(
                user_query=text,
                vector_db=vector_db,
                model_name=model_name,
                model_params=model_params,
                top_k=3
            )
        
        # Call with retry
        output = await perform_classification()
        
        # Parse the output
        if isinstance(output, dict):
            # Extract from dictionary response
            if 'choices' in output and len(output['choices']) > 0:
                if isinstance(output['choices'][0], dict) and 'message' in output['choices'][0]:
                    content = output['choices'][0]['message'].get('content', '')
                    result = content.strip()
                else:
                    result = str(output['choices'][0]).strip()
            else:
                result = str(output).strip()
        else:
            # If output is already a string
            result = str(output).strip()
        
        # Parse the result (looking for format like "DocType:True/False")
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
            'raw_result': result[:100]  # Store first 100 chars of raw result for debugging
        })
            
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
    
    # Check if we've processed a batch
    batch_count += 1
    if batch_count >= BATCH_SIZE:
        # Create intermediate results DataFrame and save
        interim_df = pd.DataFrame(classification_results)
        interim_df.to_csv(f"document_classification_interim_{i}.csv", index=False)
        print(f"Saved interim results ({len(classification_results)} documents) to document_classification_interim_{i}.csv")
        
        # Pause between batches to avoid overwhelming the server
        print(f"Pausing for {PAUSE_DURATION} seconds to avoid rate limiting...")
        time.sleep(PAUSE_DURATION)
        batch_count = 0

# Convert all results to DataFrame
results_df = pd.DataFrame(classification_results)

# Calculate accuracy metrics for non-error results
valid_results = results_df[results_df['predicted_label'] != 'ERROR']
if len(valid_results) > 0:
    label_accuracy = valid_results['correct_label'].mean() * 100
    first_page_accuracy = valid_results['correct_first_page'].mean() * 100
    
    print(f"Processed {len(valid_results)} out of {len(testdf)} documents successfully")
    print(f"Document Type Classification Accuracy: {label_accuracy:.2f}%")
    print(f"First Page Classification Accuracy: {first_page_accuracy:.2f}%")
else:
    print("No valid results to calculate accuracy")

# Generate confusion matrix for document types (only for valid predictions)
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
