# Process the entire test dataframe with the classification models

# Initialize lists to store results
doc_type_results = []
first_page_results = []
combined_results = []

# Set model parameters
model_name = "Meta-Llama-3.3-70B-Instruct"
model_params = {
    "temperature": 0.2,
    "top_p": 0.95,
    "logprobs": True,
    "prompt_type": "instruction"
}
top_k = 3

# Process each document in the test dataframe
for idx, row in testdf.iterrows():
    print(f"Processing document {idx+1}/{len(testdf)}")
    
    # Get document text
    doc_text = row["truncated_text"]
    actual_label = row["label"]
    actual_first_pg = row["first_pg"]
    
    try:
        # 1. Document type classification
        doc_type_result = await doc_classification_rag(
            user_query=doc_text,
            vector_db=vector_db,
            model_name=model_name,
            model_params=model_params,
            top_k=top_k
        )
        
        # 2. First page classification
        first_page_result = await first_page_classification_rag(
            user_query=doc_text,
            vector_db=vector_db,
            model_name=model_name,
            model_params=model_params,
            top_k=top_k
        )
        
        # 3. Combined classification
        combined_result = await combined_classification_rag(
            user_query=doc_text,
            vector_db=vector_db,
            model_name=model_name,
            model_params=model_params,
            top_k=top_k
        )
        
        # Store results along with ground truth
        doc_type_results.append({
            "doc_id": idx,
            "actual_label": actual_label,
            "predicted_label": doc_type_result,
            "text_preview": doc_text[:100] + "..."  # Store a preview of the text
        })
        
        first_page_results.append({
            "doc_id": idx,
            "actual_first_pg": actual_first_pg,
            "predicted_first_pg": first_page_result,
            "text_preview": doc_text[:100] + "..."
        })
        
        combined_results.append({
            "doc_id": idx,
            "actual_combined": f"{actual_label}:{actual_first_pg}",
            "predicted_combined": combined_result,
            "text_preview": doc_text[:100] + "..."
        })
        
    except Exception as e:
        print(f"Error processing document {idx}: {str(e)}")
        # Store error information
        error_info = {
            "doc_id": idx,
            "error": str(e),
            "text_preview": doc_text[:100] + "..."
        }
        doc_type_results.append({**error_info, "actual_label": actual_label, "predicted_label": "ERROR"})
        first_page_results.append({**error_info, "actual_first_pg": actual_first_pg, "predicted_first_pg": "ERROR"})
        combined_results.append({**error_info, "actual_combined": f"{actual_label}:{actual_first_pg}", "predicted_combined": "ERROR"})

# Convert results to dataframes for analysis
doc_type_df = pd.DataFrame(doc_type_results)
first_page_df = pd.DataFrame(first_page_results)
combined_df = pd.DataFrame(combined_results)

# Calculate accuracy metrics
doc_type_accuracy = calculate_accuracy(doc_type_df)
first_page_accuracy = calculate_accuracy(first_page_df)
combined_accuracy = calculate_accuracy(combined_df)

# Print results summary
print("\nDocument Type Classification Results:")
print(f"Accuracy: {doc_type_accuracy:.2f}")

print("\nFirst Page Classification Results:")
print(f"Accuracy: {first_page_accuracy:.2f}")

print("\nCombined Classification Results:")
print(f"Accuracy: {combined_accuracy:.2f}")

# Save results to CSV for further analysis
doc_type_df.to_csv("doc_type_classification_results.csv", index=False)
first_page_df.to_csv("first_page_classification_results.csv", index=False)
combined_df.to_csv("combined_classification_results.csv", index=False)

# Helper function to calculate accuracy
def calculate_accuracy(results_df):
    if "predicted_label" in results_df.columns:
        correct = (results_df["actual_label"] == results_df["predicted_label"]).sum()
    elif "predicted_first_pg" in results_df.columns:
        correct = (results_df["actual_first_pg"] == results_df["predicted_first_pg"]).sum()
    else:
        correct = (results_df["actual_combined"] == results_df["predicted_combined"]).sum()
    
    total = len(results_df)
    return correct / total if total > 0 else 0
