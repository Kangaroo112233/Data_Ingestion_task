import asyncio

async def batch_doc_classification_rag(testdf, vector_db, model_name, model_params, top_k=3, batch_size=5):
    """
    Processes document classification for all rows asynchronously using `asyncio.gather()`
    to handle multiple API requests concurrently.

    Args:
        testdf (pd.DataFrame): Dataframe with a 'truncated_text' column.
        vector_db: Chroma DB collection.
        model_name (str): Model name (e.g., "Meta-llama-3.3-7B-Instruct").
        model_params (dict): Model parameters like temperature, top_p.
        top_k (int): Number of retrieved chunks.
        batch_size (int): Number of requests to process in parallel.

    Returns:
        pd.DataFrame: Updated dataframe with 'pred_doc_label' column.
    """
    
    predicted_labels = []

    async def process_row(row):
        """Helper function to classify a single row."""
        user_query = row["truncated_text"]
        try:
            result = await doc_classification_rag(
                user_query=user_query,
                vector_db=vector_db,
                model_name=model_name,
                model_params=model_params,
                top_k=top_k
            )

            # Parse response
            if isinstance(result, dict) and "choices" in result:
                classification_text = result["choices"][0].get("text", "").strip()
            else:
                classification_text = str(result).strip()
            
        except Exception as e:
            classification_text = f"Error: {str(e)}"

        return classification_text

    # Process in batches
    for i in range(0, len(testdf), batch_size):
        batch = testdf.iloc[i : i + batch_size]  # Get batch
        tasks = [process_row(row) for _, row in batch.iterrows()]  # Create async tasks
        batch_results = await asyncio.gather(*tasks)  # Execute tasks concurrently
        predicted_labels.extend(batch_results)  # Store results

    # Update DataFrame
    testdf["pred_doc_label"] = predicted_labels
    return testdf


# Run the batch processing function
updated_testdf = await batch_doc_classification_rag(
    testdf=testdf,
    vector_db=collection,
    model_name="Meta-llama-3.3-7B-Instruct",
    model_params={"temperature": 0.2, "top_p": 0.95},
    top_k=3,
    batch_size=5  # Adjust based on API limits
)

# View results
print(updated_testdf[["truncated_text", "label", "pred_doc_label"]].head(10))
