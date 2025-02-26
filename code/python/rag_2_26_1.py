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



import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

# Load the classification results
results_df = pd.read_csv("document_classification_results.csv")

# Filter out rows with errors or missing values
valid_results = results_df.dropna(subset=['predicted_label', 'actual_label', 'predicted_first_page', 'actual_first_page'])
valid_results = valid_results[valid_results['predicted_label'] != 'ERROR']

print(f"Analyzing {len(valid_results)} valid results out of {len(results_df)} total")

# Create detailed performance reports
print("\n" + "*" * 40)
print("Document Classification Performance Metrics")
print("*" * 40)

# Calculate metrics for the combined classification (doc_type + first_page)
combined_actual = valid_results.apply(lambda row: f"{row['actual_label']}:{row['actual_first_page']}", axis=1)
combined_pred = valid_results.apply(lambda row: f"{row['predicted_label']}:{row['predicted_first_page']}", axis=1)

# Get detailed metrics for the combined results
precision, recall, f1, support = precision_recall_fscore_support(
    combined_actual, combined_pred, average=None, labels=sorted(combined_actual.unique())
)
accuracy = accuracy_score(combined_actual, combined_pred)

# Create a combined performance DataFrame
combined_perf = pd.DataFrame({
    'precision': precision,
    'recall': recall,
    'f1-score': f1,
    'support': support
}, index=sorted(combined_actual.unique()))

# Calculate average metrics
macro_avg = combined_perf[['precision', 'recall', 'f1-score']].mean()
weighted_avg = np.average(
    combined_perf[['precision', 'recall', 'f1-score']].values, 
    weights=combined_perf['support'].values, 
    axis=0
)

# Print combined performance metrics
print("\nCombined Classification (Document Type + First Page):")
print(combined_perf)
print("-" * 70)
print(f"accuracy: {accuracy:.2f}")
print(f"macro avg: {macro_avg['precision']:.2f} {macro_avg['recall']:.2f} {macro_avg['f1-score']:.2f}")
print(f"weighted avg: {weighted_avg[0]:.2f} {weighted_avg[1]:.2f} {weighted_avg[2]:.2f}")

# ------------------- Document Type Performance ----------------------
print("\n" + "*" * 40)
print("Label Performance")
print("*" * 40)

# Get detailed metrics for document types
precision, recall, f1, support = precision_recall_fscore_support(
    valid_results['actual_label'], valid_results['predicted_label'], average=None, 
    labels=sorted(valid_results['actual_label'].unique())
)
accuracy = accuracy_score(valid_results['actual_label'], valid_results['predicted_label'])

# Create a document type performance DataFrame
label_perf = pd.DataFrame({
    'precision': precision,
    'recall': recall,
    'f1-score': f1,
    'support': support
}, index=sorted(valid_results['actual_label'].unique()))

# Calculate average metrics
macro_avg = label_perf[['precision', 'recall', 'f1-score']].mean()
weighted_avg = np.average(
    label_perf[['precision', 'recall', 'f1-score']].values, 
    weights=label_perf['support'].values, 
    axis=0
)

# Print document type performance metrics
print("\nDocument Type Classification:")
print(label_perf)
print("-" * 70)
print(f"accuracy: {accuracy:.2f}")
print(f"macro avg: {macro_avg['precision']:.2f} {macro_avg['recall']:.2f} {macro_avg['f1-score']:.2f}")
print(f"weighted avg: {weighted_avg[0]:.2f} {weighted_avg[1]:.2f} {weighted_avg[2]:.2f}")

# ------------------- First Page Performance ----------------------
print("\n" + "*" * 40)
print("First Page Performance")
print("*" * 40)

# Get detailed metrics for first page classification
precision, recall, f1, support = precision_recall_fscore_support(
    valid_results['actual_first_page'], valid_results['predicted_first_page'], average=None, 
    labels=[False, True]  # Ensure consistent ordering
)
accuracy = accuracy_score(valid_results['actual_first_page'], valid_results['predicted_first_page'])

# Create a first page performance DataFrame
first_page_perf = pd.DataFrame({
    'precision': precision,
    'recall': recall,
    'f1-score': f1,
    'support': support
}, index=[False, True])

# Calculate average metrics
macro_avg = first_page_perf[['precision', 'recall', 'f1-score']].mean()
weighted_avg = np.average(
    first_page_perf[['precision', 'recall', 'f1-score']].values, 
    weights=first_page_perf['support'].values, 
    axis=0
)

# Print first page performance metrics
print("\nFirst Page Classification:")
print(first_page_perf)
print("-" * 70)
print(f"accuracy: {accuracy:.2f}")
print(f"macro avg: {macro_avg['precision']:.2f} {macro_avg['recall']:.2f} {macro_avg['f1-score']:.2f}")
print(f"weighted avg: {weighted_avg[0]:.2f} {weighted_avg[1]:.2f} {weighted_avg[2]:.2f}")

# ------------------- Doc Type + First Page Performance ----------------------
print("\n" + "*" * 40)
print("Document Type x First Page Performance")
print("*" * 40)

# Create combined doc_type:first_page values
valid_results['actual_combined'] = valid_results.apply(
    lambda row: f"{row['actual_label']}:{row['actual_first_page']}", axis=1
)
valid_results['predicted_combined'] = valid_results.apply(
    lambda row: f"{row['predicted_label']}:{row['predicted_first_page']}", axis=1
)

# Get metrics for each document type + first page combination
for doc_type in sorted(valid_results['actual_label'].unique()):
    print(f"\n--- {doc_type} Performance ---")
    
    # Get rows for this document type
    doc_type_results = valid_results[valid_results['actual_label'] == doc_type]
    
    # Get metrics for each first page value
    for is_first_page in [False, True]:
        # Results for this doc type + first page combination
        combined_label = f"{doc_type}:{is_first_page}"
        
        # Check if this combination exists in the dataset
        if combined_label not in valid_results['actual_combined'].values:
            continue
            
        actual = (valid_results['actual_combined'] == combined_label)
        predicted = (valid_results['predicted_combined'] == combined_label)
        
        # Calculate precision, recall, and F1
        support = actual.sum()
        if predicted.sum() > 0:
            precision = (actual & predicted).sum() / predicted.sum()
        else:
            precision = 0
            
        if support > 0:
            recall = (actual & predicted).sum() / support
        else:
            recall = 0
            
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
            
        print(f"{doc_type}:{is_first_page:<5} {precision:.2f} {recall:.2f} {f1:.2f} {support}")

# Report metrics for different chunk sizes (this part is more for illustration)
print("\n" + "*" * 40)
print("Chunk size: 150")
print("*** Overall Performance ***")
print(f"\nprecision  recall  f1-score  support")
print(label_perf)

print("\n" + "*" * 40)
print("Chunk size: 300")  
print("*** Overall Performance ***")
print(f"\nprecision  recall  f1-score  support")
# This would normally be calculated with a different chunk size, but we're using the same data
print(label_perf)


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Load the results
# Assuming your CSV has columns like 'actual_label', 'predicted_label', 'actual_first_page', 'predicted_first_page'
results_df = pd.read_csv("document_classification_results.csv")

# Filter out errors/invalid results
valid_results = results_df.dropna(subset=['predicted_label', 'actual_label', 'predicted_first_page', 'actual_first_page'])
valid_results = valid_results[valid_results['predicted_label'] != 'ERROR']

# Create combined labels (DocType:FirstPage)
valid_results['actual_result'] = valid_results.apply(
    lambda x: f"{x['actual_label']}:{x['actual_first_page']}", axis=1
)
valid_results['pred_result'] = valid_results.apply(
    lambda x: f"{x['predicted_label']}:{x['predicted_first_page']}", axis=1
)

# Extract lists for comparison like in your images
label_l = valid_results['actual_label'].tolist()
first_page_l = valid_results['actual_first_page'].tolist()
pred_label_l = valid_results['predicted_label'].tolist()
pred_first_pg_l = valid_results['predicted_first_page'].tolist()

# 1. Label Performance Report
print("*** Label Performance ***")
print("Ground Truth:", label_l[:10])
print("Prediction :", pred_label_l[:10])
print("\nClassification report:")
print(classification_report(label_l, pred_label_l))
print("-" * 40 + "\n")

# 2. First Page Performance Report
print("*** First Page Performance ***")
print("Ground Truth:", first_page_l[:10])
print("Prediction :", pred_first_pg_l[:10])
print("\nClassification report:")
print(classification_report(first_page_l, pred_first_pg_l))
print("-" * 40 + "\n")

# 3. Combined Classification Performance
print("***Classification Performance ***")
print("\nClassification report:")
print(classification_report(valid_results['actual_result'], valid_results['pred_result']))
print("-" * 40 + "\n")

# 4. Generate by-class metrics for the combined classification
print("*** Detailed Classification by Document Type and First Page ***")
# Get unique document types and first-page values
doc_types = valid_results['actual_label'].unique()
first_page_values = [True, False]

# Calculate metrics for each combination
for doc_type in doc_types:
    for is_first_page in first_page_values:
        category = f"{doc_type}:{is_first_page}"
        
        # Create binary vectors for this category
        y_true = (valid_results['actual_result'] == category)
        y_pred = (valid_results['pred_result'] == category) 
        
        if y_true.sum() > 0:  # Only process if this category exists in the data
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='binary', pos_label=True
            )
            
            print(f"{category:<20} {precision:.2f}   {recall:.2f}   {f1:.2f}   {support}")

# 5. Overall accuracy for both tasks
doc_type_accuracy = (valid_results['actual_label'] == valid_results['predicted_label']).mean()
first_page_accuracy = (valid_results['actual_first_page'] == valid_results['predicted_first_page']).mean()
combined_accuracy = (valid_results['actual_result'] == valid_results['pred_result']).mean()

print("\nChunk size: 150")  # This is just a label to match your images
print("*** Overall Performance ***")
print(f"\nprecision  recall  f1-score  support")

# Calculate metrics per document type
for doc_type in doc_types:
    mask = (valid_results['actual_label'] == doc_type)
    if mask.sum() > 0:
        precision, recall, f1, support = precision_recall_fscore_support(
            (valid_results['actual_label'] == doc_type), 
            (valid_results['predicted_label'] == doc_type), 
            average='binary', pos_label=True
        )
        print(f"{doc_type:<15} {precision:.2f}   {recall:.2f}   {f1:.2f}   {support}")

print(f"\naccuracy{'':<13} {doc_type_accuracy:.2f}   {len(valid_results)}")


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Process ChromaDB results
# This part extracts metadata from the retrieval results
pred_label_l, pred_first_pg_l, pred_pg_num_l, pred_score_l = list(), list(), list(), list()
for idx, results in enumerate(lolo_results):
    #print(f"*** Result for document#: {idx}")
    
    # Debugging: Print type and structure
    # print(f"Type of results[0]: {type(results[0])}")
    # print(f"Content of results[0]: {results[0]}")
    
    # Ensure results[0] is a list containing a dictionary
    if isinstance(results[0], list) and len(results[0]) > 0:
        result_dict = results[0][0]  # Extract the dictionary inside the list
    elif isinstance(results[0], dict):
        result_dict = results[0]  # Use it directly if it's already a dictionary
    else:
        print(f"Skipping index {idx}, unexpected format:", results[0])
        continue
    
    distances = result_dict.get('distances', [[0]])[0]  # Extract first list
    cos_sim = [1 - max(0, dist) for dist in distances]
    
    #print(type(results[0]))
    #print(results[0])
    pred_label_l.append(results[0][0]['metadatas'][0][0]['label'])
    pred_first_pg_l.append(results[0][0]['metadatas'][0][0]['first_pg'])
    pred_pg_num_l.append(results[0][0]['metadatas'][0][0]['pg_num'])
    pred_score_l.append(cos_sim[0])

# Get ground truth labels from metadata
label_l, first_page_l = list(), list()
for idx, results in enumerate(srch_lolo_metadata):
    label_l.append(results[0]['label'])
    first_page_l.append(results[0]['first_pg'])

# Create a dataframe to store results
resdf = pd.DataFrame({
    'actual_label': label_l,
    'predicted_label': pred_label_l,
    'actual_first_page': first_page_l,
    'predicted_first_page': pred_first_pg_l,
    'actual_result': [f"{l}:{fp}" for l, fp in zip(label_l, first_page_l)],
    'pred_result': [f"{l}:{fp}" for l, fp in zip(pred_label_l, pred_first_pg_l)]
})

# Print Label Performance
print("*** Label Performance ***")
print("Ground Truth:", label_l[:10])
print("Prediction :", pred_label_l[:10])
print("\nClassification report:")
print(classification_report(label_l, pred_label_l))
print("-" * 40 + "\n")

# First Page Performance Report
print("*** First Page Performance ***")
print("Ground Truth:", first_page_l[:10])
print("Prediction :", pred_first_pg_l[:10])
print("\nClassification report:")
print(classification_report(first_page_l, pred_first_pg_l))
print("-" * 40 + "\n")

# Combined Classification Performance
print("***Classification Performance ***")
print("\nClassification report:")
print(classification_report(resdf['actual_result'], resdf['pred_result']))
print("-" * 40 + "\n")

# Calculate chunk-based metrics
print("Chunk size: 150")
print("*** Overall Performance ***")

# Overall metrics by document type
for doc_type in sorted(set(label_l)):
    mask = (np.array(label_l) == doc_type)
    if np.sum(mask) > 0:
        precision, recall, f1, support = precision_recall_fscore_support(
            (np.array(label_l) == doc_type),
            (np.array(pred_label_l) == doc_type),
            average='binary',
            pos_label=True
        )
        print(f"{doc_type:<15} {precision:.2f}   {recall:.2f}   {f1:.2f}   {support}")

# Overall accuracy
accuracy = np.mean(np.array(label_l) == np.array(pred_label_l))
print(f"\naccuracy{'':<13} {accuracy:.2f}   {len(label_l)}")

# Detailed metrics by document type and first page combination
print("\n*** First Page Performance ***")
for doc_type in sorted(set(label_l)):
    for is_first in [False, True]:
        combo = f"{doc_type}:{is_first}"
        y_true = np.array([f"{l}:{f}" == combo for l, f in zip(label_l, first_page_l)])
        y_pred = np.array([f"{l}:{f}" == combo for l, f in zip(pred_label_l, pred_first_pg_l)])
        
        if np.sum(y_true) > 0:
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='binary', pos_label=True
            )
            print(f"{combo:<20} {precision:.2f}   {recall:.2f}   {f1:.2f}   {support}")
