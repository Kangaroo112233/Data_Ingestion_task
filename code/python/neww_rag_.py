import asyncio
import pandas as pd
from tqdm import tqdm
import json

async def batch_classify_documents(test_df, vector_db, model_name, batch_size=10):
    """
    Apply classification to all documents in the test dataframe.
    
    Args:
        test_df (pd.DataFrame): Test dataframe containing documents to classify
        vector_db: ChromaDB collection for document retrieval
        model_name (str): Name of the LLM model to use
        batch_size (int): Number of documents to process in each batch
        
    Returns:
        pd.DataFrame: Original dataframe with added prediction columns
    """
    # Create a copy of the dataframe to add predictions
    results_df = test_df.copy()
    
    # Add columns for predictions
    results_df['predicted_label'] = None
    results_df['predicted_first_page'] = None
    results_df['combined_prediction'] = None
    
    # Set up model parameters
    model_params = {
        "temperature": 0.2,
        "top_p": 0.95,
        "logprobs": True,
        "prompt_type": "instruction"
    }
    
    # Process in batches to avoid overwhelming the API
    for i in tqdm(range(0, len(test_df), batch_size), desc="Processing batches"):
        batch = test_df.iloc[i:i+batch_size]
        
        # Create tasks for all documents in the batch
        doc_class_tasks = []
        fp_class_tasks = []
        combined_tasks = []
        
        for idx, row in batch.iterrows():
            doc_text = row["truncated_text"]
            
            # Create async tasks for each classification type
            doc_task = doc_classification_rag(
                user_query=doc_text,
                vector_db=vector_db,
                model_name=model_name,
                model_params=model_params,
                top_k=3
            )
            doc_class_tasks.append((idx, doc_task))
            
            fp_task = first_page_classification_rag(
                user_query=doc_text,
                vector_db=vector_db,
                model_name=model_name,
                model_params=model_params,
                top_k=3
            )
            fp_class_tasks.append((idx, fp_task))
            
            combined_task = combined_classification_rag(
                user_query=doc_text,
                vector_db=vector_db,
                model_name=model_name,
                model_params=model_params,
                top_k=3
            )
            combined_tasks.append((idx, combined_task))
        
        # Execute all tasks for this batch concurrently
        # Document classification results
        for idx, task in doc_class_tasks:
            try:
                result = await task
                # Extract the classification from the result
                label = extract_doc_label(result)
                results_df.at[idx, 'predicted_label'] = label
            except Exception as e:
                print(f"Error in document classification for index {idx}: {e}")
                results_df.at[idx, 'predicted_label'] = "ERROR"
        
        # First page classification results
        for idx, task in fp_class_tasks:
            try:
                result = await task
                # Extract the first page classification (True/False)
                is_first_page = extract_first_page(result)
                results_df.at[idx, 'predicted_first_page'] = is_first_page
            except Exception as e:
                print(f"Error in first page classification for index {idx}: {e}")
                results_df.at[idx, 'predicted_first_page'] = "ERROR"
        
        # Combined classification results
        for idx, task in combined_tasks:
            try:
                result = await task
                # Extract the combined classification
                combined = extract_combined_result(result)
                results_df.at[idx, 'combined_prediction'] = combined
            except Exception as e:
                print(f"Error in combined classification for index {idx}: {e}")
                results_df.at[idx, 'combined_prediction'] = "ERROR"
                
        # Pause between batches to avoid rate limiting
        if i + batch_size < len(test_df):
            await asyncio.sleep(1)
            
    return results_df

def extract_doc_label(result):
    """Extract document label from the API response"""
    try:
        content = result["choices"][0]["message"]["content"]
        # Find common document types in the response
        if "Bank Statement" in content or "bank statement" in content.lower():
            return "Bank Statement"
        elif "Paystub" in content or "paystub" in content.lower() or "pay stub" in content.lower():
            return "Paystub"
        elif "W2" in content or "w-2" in content.lower():
            return "W2"
        elif "Other" in content:
            return "Other"
        else:
            # Try to find the answer after "Answer:" pattern
            if "Answer:" in content:
                answer = content.split("Answer:")[1].strip()
                return answer
            return content  # Return the full content if parsing fails
    except Exception as e:
        print(f"Error parsing document label: {e}")
        return "ERROR_PARSING"

def extract_first_page(result):
    """Extract first page classification from the API response"""
    try:
        content = result["choices"][0]["message"]["content"].lower()
        if "true" in content:
            return True
        elif "false" in content:
            return False
        else:
            return "UNKNOWN"
    except Exception as e:
        print(f"Error parsing first page result: {e}")
        return "ERROR_PARSING"

def extract_combined_result(result):
    """Extract combined classification from the API response"""
    try:
        content = result["choices"][0]["message"]["content"]
        return content.strip()
    except Exception as e:
        print(f"Error parsing combined result: {e}")
        return "ERROR_PARSING"

async def evaluate_performance(results_df):
    """
    Evaluate the performance of the classification models.
    
    Args:
        results_df (pd.DataFrame): DataFrame with predictions and ground truth
        
    Returns:
        dict: Performance metrics
    """
    # Document classification accuracy
    doc_correct = (results_df['label'] == results_df['predicted_label']).sum()
    doc_total = len(results_df)
    doc_accuracy = doc_correct / doc_total if doc_total > 0 else 0
    
    # First page classification accuracy
    fp_correct = (results_df['first_pg'] == results_df['predicted_first_page']).sum()
    fp_total = len(results_df)
    fp_accuracy = fp_correct / fp_total if fp_total > 0 else 0
    
    # Combined accuracy calculation
    # This assumes combined_prediction is in format "DocType: FirstPage"
    combined_correct = 0
    for idx, row in results_df.iterrows():
        try:
            combined_pred = row['combined_prediction']
            if ":" in combined_pred:
                doc_type, is_first = combined_pred.split(":")
                doc_type = doc_type.strip()
                is_first_bool = "true" in is_first.lower()
                
                if (doc_type == row['label'] and is_first_bool == row['first_pg']):
                    combined_correct += 1
        except:
            pass
    
    combined_accuracy = combined_correct / len(results_df) if len(results_df) > 0 else 0
    
    # Per-class performance
    doc_class_metrics = {}
    for label in results_df['label'].unique():
        class_df = results_df[results_df['label'] == label]
        correct = (class_df['label'] == class_df['predicted_label']).sum()
        total = len(class_df)
        accuracy = correct / total if total > 0 else 0
        doc_class_metrics[label] = {
            'accuracy': accuracy,
            'samples': total
        }
    
    metrics = {
        'document_classification_accuracy': doc_accuracy,
        'first_page_classification_accuracy': fp_accuracy,
        'combined_classification_accuracy': combined_accuracy,
        'per_class_metrics': doc_class_metrics,
        'total_samples': len(results_df)
    }
    
    return metrics

# Main execution function
async def main():
    # Assuming the necessary variables are already defined:
    # testdf, vector_db, model_name
    
    print(f"Processing {len(testdf)} documents...")
    
    # Process a smaller subset first if you want to test
    # test_subset = testdf.sample(10)  # Uncomment to use a random sample
    
    # Or use the full test set
    results_df = await batch_classify_documents(
        testdf, 
        vector_db=vector_db,
        model_name="Meta-Llama-3.3-70B-Instruct",
        batch_size=5  # Adjust based on API rate limits
    )
    
    # Save results to CSV
    results_df.to_csv("document_classification_results.csv", index=False)
    print("Results saved to document_classification_results.csv")
    
    # Evaluate performance
    metrics = await evaluate_performance(results_df)
    
    # Print performance metrics
    print("\n=== Classification Performance ===")
    print(f"Document classification accuracy: {metrics['document_classification_accuracy']:.4f}")
    print(f"First page classification accuracy: {metrics['first_page_classification_accuracy']:.4f}")
    print(f"Combined classification accuracy: {metrics['combined_classification_accuracy']:.4f}")
    
    print("\n=== Per-Class Performance ===")
    for label, data in metrics['per_class_metrics'].items():
        print(f"{label}: Accuracy = {data['accuracy']:.4f} ({data['samples']} samples)")
    
    # Save metrics to JSON
    with open("classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return results_df, metrics

# To run the code:
# results, metrics = await main()
