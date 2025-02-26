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
