# Function to process the entire test dataset
async def process_all_test_documents(testdf, vector_db, model_name="Meta-Llama-3.3-70B-Instruct"):
    # Create result columns
    results_df = testdf.copy()
    results_df["predicted_label"] = ""
    results_df["predicted_first_page"] = ""
    results_df["combined_result"] = ""
    results_df["is_correct_label"] = False
    results_df["is_correct_first_page"] = False
    
    # Set model parameters
    model_params = {
        "temperature": 0.2,
        "top_p": 0.95
    }
    
    # Process each document in the test set
    print(f"Processing {len(testdf)} documents...")
    
    for idx, row in testdf.iterrows():
        if idx % 10 == 0:
            print(f"Processing document {idx}/{len(testdf)}...")
        
        doc_text = row["truncated_text"]
        true_label = row["label"]
        true_first_page = row["first_pg"]
        
        # Get combined classification (more efficient than getting both separately)
        combined_result = await combined_classification_rag(
            user_query=doc_text,
            vector_db=vector_db,
            model_name=model_name,
            model_params=model_params,
            top_k=3
        )
        
        # Parse combined result (format: "DocumentType:FirstPageStatus")
        try:
            parts = combined_result.split(":")
            predicted_label = parts[0].strip()
            predicted_first_page = parts[1].strip().lower() == "true"
            
            # Store results
            results_df.at[idx, "predicted_label"] = predicted_label
            results_df.at[idx, "predicted_first_page"] = predicted_first_page
            results_df.at[idx, "combined_result"] = combined_result
            
            # Check if predictions are correct
            results_df.at[idx, "is_correct_label"] = predicted_label.lower() == true_label.lower()
            results_df.at[idx, "is_correct_first_page"] = predicted_first_page == true_first_page
            
        except Exception as e:
            print(f"Error processing document {idx}: {e}")
            results_df.at[idx, "combined_result"] = f"Error: {str(e)}"
    
    return results_df

# Run the processing function
async def main():
    # Assuming vector_db is already initialized
    print("Starting document classification...")
    results_df = await process_all_test_documents(testdf, vector_db)
    
    # Calculate and display accuracy metrics
    label_accuracy = results_df["is_correct_label"].mean() * 100
    first_page_accuracy = results_df["is_correct_first_page"].mean() * 100
    overall_accuracy = (results_df["is_correct_label"] & results_df["is_correct_first_page"]).mean() * 100
    
    print(f"\nClassification Results:")
    print(f"Document Type Accuracy: {label_accuracy:.2f}%")
    print(f"First Page Detection Accuracy: {first_page_accuracy:.2f}%")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    
    # Display breakdown by document type
    print("\nAccuracy by Document Type:")
    for doc_type in testdf["label"].unique():
        type_mask = testdf["label"] == doc_type
        type_accuracy = results_df.loc[type_mask, "is_correct_label"].mean() * 100
        print(f"{doc_type}: {type_accuracy:.2f}% ({results_df.loc[type_mask, 'is_correct_label'].sum()} / {type_mask.sum()})")
    
    # Save results to CSV
    results_df.to_csv("classification_results.csv", index=False)
    print("Results saved to classification_results.csv")
    
    return results_df

# Execute the main function
import asyncio
resultdf = asyncio.run(main())
