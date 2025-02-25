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





import asyncio
import pandas as pd
from tqdm import tqdm
import json

async def batch_classify_documents(test_df, vector_db, model_name, batch_size=10, max_retries=3, retry_delay=2):
    """
    Apply classification to all documents in the test dataframe with error handling and retries.
    
    Args:
        test_df (pd.DataFrame): Test dataframe containing documents to classify
        vector_db: ChromaDB collection for document retrieval
        model_name (str): Name of the LLM model to use
        batch_size (int): Number of documents to process in each batch
        max_retries (int): Maximum number of retry attempts for failed requests
        retry_delay (int): Delay in seconds between retries
        
    Returns:
        pd.DataFrame: Original dataframe with added prediction columns
    """
    # Create a copy of the dataframe to add predictions
    results_df = test_df.copy()
    
    # Add columns for predictions
    results_df['predicted_label'] = None
    results_df['predicted_first_page'] = None
    results_df['combined_prediction'] = None
    results_df['error_messages'] = None
    
    # Set up model parameters - remove logprobs which might be causing issues
    model_params = {
        "temperature": 0.2,
        "top_p": 0.95,
        "prompt_type": "instruction"
    }
    
    # Process in batches to avoid overwhelming the API
    for i in tqdm(range(0, len(test_df), batch_size), desc="Processing batches"):
        batch = test_df.iloc[i:i+batch_size]
        
        for idx, row in batch.iterrows():
            doc_text = row["truncated_text"]
            errors = []
            
            # Document classification with retries
            for retry in range(max_retries):
                try:
                    # Use a smaller timeout for each request
                    doc_result = await doc_classification_rag(
                        user_query=doc_text,
                        vector_db=vector_db,
                        model_name=model_name,
                        model_params=model_params,
                        top_k=3
                    )
                    
                    # Check if the response is a string (error message) or dictionary (success)
                    if isinstance(doc_result, str):
                        print(f"API returned string error: {doc_result}")
                        errors.append(f"Doc classification error: {doc_result}")
                        results_df.at[idx, 'predicted_label'] = "API_ERROR"
                    else:
                        # Try to parse as normal
                        try:
                            label = extract_doc_label(doc_result)
                            results_df.at[idx, 'predicted_label'] = label
                            break  # Success, exit retry loop
                        except Exception as e:
                            print(f"Error parsing document label: {e}")
                            errors.append(f"Doc parsing error: {str(e)}")
                            results_df.at[idx, 'predicted_label'] = "PARSE_ERROR"
                
                except Exception as e:
                    print(f"Request error for document classification: {e}")
                    errors.append(f"Doc request error: {str(e)}")
                    results_df.at[idx, 'predicted_label'] = "REQUEST_ERROR"
                
                # Wait before retry if this wasn't the last attempt
                if retry < max_retries - 1:
                    await asyncio.sleep(retry_delay)
            
            # First page classification with retries
            for retry in range(max_retries):
                try:
                    fp_result = await first_page_classification_rag(
                        user_query=doc_text,
                        vector_db=vector_db,
                        model_name=model_name,
                        model_params=model_params,
                        top_k=3
                    )
                    
                    if isinstance(fp_result, str):
                        print(f"API returned string error: {fp_result}")
                        errors.append(f"First page error: {fp_result}")
                        results_df.at[idx, 'predicted_first_page'] = "API_ERROR"
                    else:
                        try:
                            is_first_page = extract_first_page(fp_result)
                            results_df.at[idx, 'predicted_first_page'] = is_first_page
                            break  # Success, exit retry loop
                        except Exception as e:
                            print(f"Error parsing first page result: {e}")
                            errors.append(f"FP parsing error: {str(e)}")
                            results_df.at[idx, 'predicted_first_page'] = "PARSE_ERROR"
                
                except Exception as e:
                    print(f"Request error for first page classification: {e}")
                    errors.append(f"FP request error: {str(e)}")
                    results_df.at[idx, 'predicted_first_page'] = "REQUEST_ERROR"
                
                if retry < max_retries - 1:
                    await asyncio.sleep(retry_delay)
            
            # Combined classification with retries
            for retry in range(max_retries):
                try:
                    combined_result = await combined_classification_rag(
                        user_query=doc_text,
                        vector_db=vector_db,
                        model_name=model_name,
                        model_params=model_params,
                        top_k=3
                    )
                    
                    if isinstance(combined_result, str):
                        print(f"API returned string error: {combined_result}")
                        errors.append(f"Combined error: {combined_result}")
                        results_df.at[idx, 'combined_prediction'] = "API_ERROR"
                    else:
                        try:
                            combined = extract_combined_result(combined_result)
                            results_df.at[idx, 'combined_prediction'] = combined
                            break  # Success, exit retry loop
                        except Exception as e:
                            print(f"Error parsing combined result: {e}")
                            errors.append(f"Combined parsing error: {str(e)}")
                            results_df.at[idx, 'combined_prediction'] = "PARSE_ERROR"
                
                except Exception as e:
                    print(f"Request error for combined classification: {e}")
                    errors.append(f"Combined request error: {str(e)}")
                    results_df.at[idx, 'combined_prediction'] = "REQUEST_ERROR"
                
                if retry < max_retries - 1:
                    await asyncio.sleep(retry_delay)
            
            # Store any error messages
            if errors:
                results_df.at[idx, 'error_messages'] = "; ".join(errors)
            
            # Add a small delay between documents to avoid rate limiting
            await asyncio.sleep(0.5)
        
        # Save intermediate results after each batch
        results_df.to_csv(f"document_classification_results_batch_{i}.csv", index=False)
        
        # Pause between batches to avoid rate limiting
        if i + batch_size < len(test_df):
            await asyncio.sleep(2)
            
    return results_df

def extract_doc_label(result):
    """Extract document label from the API response with robust error handling"""
    try:
        # Check if the result is a dictionary and has the expected structure
        if not isinstance(result, dict) or "choices" not in result:
            if hasattr(result, "status_code"):
                return f"HTTP_ERROR_{result.status_code}"
            return "INVALID_RESPONSE_FORMAT"
        
        # Extract content safely
        choices = result.get("choices", [])
        if not choices:
            return "NO_CHOICES_IN_RESPONSE"
        
        first_choice = choices[0]
        if not isinstance(first_choice, dict) or "message" not in first_choice:
            return "INVALID_CHOICE_FORMAT"
        
        message = first_choice.get("message", {})
        if not isinstance(message, dict) or "content" not in message:
            return "INVALID_MESSAGE_FORMAT"
        
        content = message.get("content", "")
        if not content:
            return "EMPTY_CONTENT"
        
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
        return f"ERROR_PARSING: {str(e)}"

def extract_first_page(result):
    """Extract first page classification from the API response with robust error handling"""
    try:
        # Check if the result is a dictionary and has the expected structure
        if not isinstance(result, dict) or "choices" not in result:
            if hasattr(result, "status_code"):
                return f"HTTP_ERROR_{result.status_code}"
            return "INVALID_RESPONSE_FORMAT"
        
        # Extract content safely
        choices = result.get("choices", [])
        if not choices:
            return "NO_CHOICES_IN_RESPONSE"
        
        first_choice = choices[0]
        if not isinstance(first_choice, dict) or "message" not in first_choice:
            return "INVALID_CHOICE_FORMAT"
        
        message = first_choice.get("message", {})
        if not isinstance(message, dict) or "content" not in message:
            return "INVALID_MESSAGE_FORMAT"
        
        content = message.get("content", "").lower()
        if not content:
            return "EMPTY_CONTENT"
        
        if "true" in content:
            return True
        elif "false" in content:
            return False
        else:
            return "UNKNOWN_BOOLEAN_VALUE"
    except Exception as e:
        print(f"Error parsing first page result: {e}")
        return f"ERROR_PARSING: {str(e)}"

def extract_combined_result(result):
    """Extract combined classification from the API response with robust error handling"""
    try:
        # Check if the result is a dictionary and has the expected structure
        if not isinstance(result, dict) or "choices" not in result:
            if hasattr(result, "status_code"):
                return f"HTTP_ERROR_{result.status_code}"
            return "INVALID_RESPONSE_FORMAT"
        
        # Extract content safely
        choices = result.get("choices", [])
        if not choices:
            return "NO_CHOICES_IN_RESPONSE"
        
        first_choice = choices[0]
        if not isinstance(first_choice, dict) or "message" not in first_choice:
            return "INVALID_CHOICE_FORMAT"
        
        message = first_choice.get("message", {})
        if not isinstance(message, dict) or "content" not in message:
            return "INVALID_MESSAGE_FORMAT"
        
        content = message.get("content", "")
        if not content:
            return "EMPTY_CONTENT"
        
        return content.strip()
    except Exception as e:
        print(f"Error parsing combined result: {e}")
        return f"ERROR_PARSING: {str(e)}"

async def evaluate_performance(results_df):
    """
    Evaluate the performance of the classification models with improved error handling.
    
    Args:
        results_df (pd.DataFrame): DataFrame with predictions and ground truth
        
    Returns:
        dict: Performance metrics
    """
    # Filter out error values for more accurate metrics
    valid_doc_df = results_df[~results_df['predicted_label'].isin([
        "API_ERROR", "PARSE_ERROR", "REQUEST_ERROR", "INVALID_RESPONSE_FORMAT", 
        "NO_CHOICES_IN_RESPONSE", "INVALID_CHOICE_FORMAT", "INVALID_MESSAGE_FORMAT",
        "EMPTY_CONTENT", "ERROR_PARSING", None
    ])]
    
    valid_fp_df = results_df[~pd.isna(results_df['predicted_first_page']) & 
                             ~results_df['predicted_first_page'].astype(str).str.contains("ERROR|INVALID|UNKNOWN")]
    
    valid_combined_df = results_df[~pd.isna(results_df['combined_prediction']) & 
                                   ~results_df['combined_prediction'].astype(str).str.contains("ERROR|INVALID")]
    
    # Document classification accuracy (only on valid predictions)
    doc_correct = (valid_doc_df['label'] == valid_doc_df['predicted_label']).sum()
    doc_total = len(valid_doc_df)
    doc_accuracy = doc_correct / doc_total if doc_total > 0 else 0
    
    # First page classification accuracy (only on valid predictions)
    # Convert predicted_first_page to boolean for comparison if it's a string "True" or "False"
    valid_fp_df = valid_fp_df.copy()
    valid_fp_df['predicted_first_page_bool'] = valid_fp_df['predicted_first_page'].apply(
        lambda x: True if str(x).lower() == 'true' else (False if str(x).lower() == 'false' else x)
    )
    
    fp_correct = (valid_fp_df['first_pg'] == valid_fp_df['predicted_first_page_bool']).sum()
    fp_total = len(valid_fp_df)
    fp_accuracy = fp_correct / fp_total if fp_total > 0 else 0
    
    # Combined accuracy calculation (only on valid predictions)
    combined_correct = 0
    parsed_combined = 0
    
    for idx, row in valid_combined_df.iterrows():
        try:
            combined_pred = row['combined_prediction']
            # Different formats might be returned
            if ":" in combined_pred:
                # Format like "

# Main execution function
async def main():
    """
    Main execution function with improved error handling and progress tracking
    """
    # Assuming the necessary variables are already defined:
    # testdf, vector_db, model_name
    
    print(f"Processing {len(testdf)} documents...")
    
    # Process a small subset first to test the API
    small_test = True
    
    if small_test:
        print("TESTING MODE: Processing only 3 documents first to test the API...")
        # Take 3 random samples, one from each major document type if possible
        sample_indices = []
        for label in ["Bank Statement", "Paystub", "W2"]:
            filtered = testdf[testdf["label"] == label]
            if not filtered.empty:
                sample_indices.append(filtered.index[0])
        
        # If we don't have all three types, add random samples to get to 3
        while len(sample_indices) < 3 and len(testdf) > len(sample_indices):
            # Get random indices not already in sample_indices
            remaining_indices = [idx for idx in testdf.index if idx not in sample_indices]
            if remaining_indices:
                sample_indices.append(np.random.choice(remaining_indices))
        
        test_subset = testdf.loc[sample_indices]
        print(f"Test subset contains {len(test_subset)} documents with labels: {test_subset['label'].tolist()}")
        
        # Process the small subset with single document batch size for debugging
        results_df = await batch_classify_documents(
            test_subset, 
            vector_db=vector_db,
            model_name="Meta-Llama-3.3-70B-Instruct",
            batch_size=1,  # Process one at a time for testing
            max_retries=2,
            retry_delay=1
        )
        
        # Save test results
        results_df.to_csv("test_classification_results.csv", index=False)
        print("Test results saved to test_classification_results.csv")
        
        # Check for any errors
        error_count = results_df['error_messages'].notna().sum()
        if error_count > 0:
            print(f"WARNING: {error_count} out of {len(results_df)} documents had errors during test run.")
            print("Example error messages:")
            for msg in results_df['error_messages'].dropna().head(3):
                print(f"- {msg}")
                
            # Ask whether to continue with full dataset
            print("\nDo you want to continue with the full dataset? [y/N]")
            # This would typically wait for user input, but in this script we'll continue anyway
            print("Continuing with full dataset processing...")
        else:
            print("Test run completed successfully with no errors.")
    
    # Process the full dataset with appropriate batch size
    print(f"\nProcessing full dataset with {len(testdf)} documents...")
    try:
        results_df = await batch_classify_documents(
            testdf, 
            vector_db=vector_db,
            model_name="Meta-Llama-3.3-70B-Instruct",
            batch_size=3,  # Reduced batch size to minimize API errors
            max_retries=3,
            retry_delay=2
        )
        
        # Save full results to CSV
        results_df.to_csv("document_classification_results.csv", index=False)
        print("Results saved to document_classification_results.csv")
        
        # Evaluate performance only on non-error results
        valid_results = results_df[
            ~results_df['predicted_label'].isin(["API_ERROR", "PARSE_ERROR", "REQUEST_ERROR", None]) &
            ~results_df['predicted_first_page'].isin(["API_ERROR", "PARSE_ERROR", "REQUEST_ERROR", None])
        ]
        
        if len(valid_results) > 0:
            print(f"Evaluating performance on {len(valid_results)} valid results out of {len(results_df)} total")
            metrics = await evaluate_performance(valid_results)
            
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
        else:
            print("No valid results to evaluate performance on.")
            metrics = None
            
        # Error analysis
        error_count = results_df['error_messages'].notna().sum()
        print(f"\n=== Error Analysis ===")
        print(f"Total errors: {error_count} out of {len(results_df)} documents ({error_count/len(results_df)*100:.2f}%)")
        
        # Group errors by type
        error_types = {
            "API_ERROR": results_df['predicted_label'].eq("API_ERROR").sum(),
            "PARSE_ERROR": results_df['predicted_label'].eq("PARSE_ERROR").sum(),
            "REQUEST_ERROR": results_df['predicted_label'].eq("REQUEST_ERROR").sum(),
        }
        
        print("Error types:")
        for error_type, count in error_types.items():
            print(f"- {error_type}: {count} ({count/len(results_df)*100:.2f}%)")
        
        return results_df, metrics
    
    except Exception as e:
        print(f"Critical error in main execution: {e}")
        # Save partial results if available
        try:
            if 'results_df' in locals() and results_df is not None:
                results_df.to_csv("partial_results_emergency_save.csv", index=False)
                print("Partial results saved to partial_results_emergency_save.csv")
        except:
            print("Could not save partial results")
        
        raise

# To run the code:
# results, metrics = await main()
