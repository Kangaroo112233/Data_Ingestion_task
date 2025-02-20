async def process_full_testdf(testdf, vector_db):
    """
    Process the entire test dataframe through various classification functions.
    
    Parameters:
    testdf (pd.DataFrame): The test dataframe containing document records
    vector_db: Vector database for semantic search
    
    Returns:
    pd.DataFrame: Original dataframe with classification results
    """
    # Create a copy to store results
    results_df = testdf.copy()
    
    # Add columns to store classification results
    results_df['doc_classification'] = None
    results_df['first_page_classification'] = None
    results_df['combined_classification'] = None
    
    # Process each document in the dataframe
    for idx, row in testdf.iterrows():
        # Extract document text
        test_doc_text = row.get("truncated_text", "")
        
        # If text is empty, skip this document
        if not test_doc_text:
            continue
            
        # Perform document classification
        doc_result = await doc_classification_rag(
            user_query=test_doc_text,
            vector_db=vector_db,
            model_name="Meta-Llama-3.3-70B-Instruct",
            model_params={
                "temperature": 0.2,
                "top_p": 0.95,
                "logprobs": True,
                "prompt_type": "instruction"
            },
            top_k=3
        )
        
        # Perform first-page classification
        first_page_result = await first_page_classification_rag(
            user_query=test_doc_text,
            vector_db=vector_db,
            model_name="Meta-Llama-3.3-70B-Instruct",
            model_params={
                "temperature": 0.2,
                "top_p": 0.95
            },
            top_k=3
        )
        
        # Perform combined classification
        combined_result = await combined_classification_rag(
            user_query=test_doc_text,
            vector_db=vector_db,
            model_name="Meta-Llama-3.3-70B-Instruct",
            model_params={
                "temperature": 0.2,
                "top_p": 0.95
            },
            top_k=3
        )
        
        # Extract classification results
        try:
            doc_class = doc_result.get("choices", [{}])[0].get("message", {}).get("content", "Unknown")
            first_page_class = first_page_result.get("choices", [{}])[0].get("message", {}).get("content", "Unknown")
            combined_class = combined_result.get("choices", [{}])[0].get("message", {}).get("content", "Unknown")
            
            # Store results in dataframe
            results_df.at[idx, 'doc_classification'] = doc_class
            results_df.at[idx, 'first_page_classification'] = first_page_class
            results_df.at[idx, 'combined_classification'] = combined_class
            
            # Print progress
            print(f"Processed document {idx+1}/{len(testdf)}")
            
        except Exception as e:
            print(f"Error processing document {idx}: {str(e)}")
            continue
    
    return results_df

# Function to generate a summary report of classifications
def generate_classification_summary(results_df):
    """
    Generate a summary of classification results.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame with classification results
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        "total_documents": len(results_df),
        "doc_classification_counts": results_df['doc_classification'].value_counts().to_dict(),
        "first_page_classification_counts": results_df['first_page_classification'].value_counts().to_dict(),
        "combined_classification_counts": results_df['combined_classification'].value_counts().to_dict(),
        "agreement_percentage": calculate_agreement_percentage(results_df)
    }
    
    return summary

def calculate_agreement_percentage(results_df):
    """
    Calculate the percentage of documents where all classifications agree.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame with classification results
    
    Returns:
    float: Agreement percentage
    """
    # Count documents where all three classifications match
    agreement_count = 0
    
    for _, row in results_df.iterrows():
        if row['doc_classification'] == row['first_page_classification'] == row['combined_classification']:
            agreement_count += 1
    
    # Calculate percentage
    agreement_percentage = (agreement_count / len(results_df)) * 100 if len(results_df) > 0 else 0
    
    return agreement_percentage

# Example usage
async def main():
    # Load your test dataframe
    # This is a placeholder - replace with your actual data loading code
    # testdf = pd.read_csv('your_test_data.csv')
    
    # For demonstration, create a sample dataframe
    testdf = pd.DataFrame({
        'iloc': list(range(10)),
        'truncated_text': ["Sample paystub text for document " + str(i) for i in range(10)],
        'label': ["Paystub" for _ in range(10)],
        'first_pg': [True for _ in range(10)]
    })
    
    # Process the full test dataframe
    results = await process_full_testdf(testdf, vector_db)
    
    # Generate and print summary
    summary = generate_classification_summary(results)
    print("\nClassification Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Export results to CSV
    results.to_csv('classification_results.csv', index=False)
    print("\nResults exported to classification_results.csv")
    
    return results
