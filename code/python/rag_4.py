async def batch_doc_classification_rag(testdf, vector_db, model_name, model_params, top_k=3):
    """
    Iterates over each row in 'testdf', calls doc_classification_rag asynchronously,
    and stores the predicted document type in a new column 'pred_doc_label'.
    
    Args:
        testdf (pd.DataFrame): Must have a 'truncated_text' column.
        vector_db: Your Chroma DB collection.
        model_name (str): Name of the remote model (e.g., "Meta-llama-3.3-7B-Instruct").
        model_params (dict): Generation parameters (temperature, top_p, etc.).
        top_k (int): Number of chunks to retrieve from Chroma DB.
    
    Returns:
        pd.DataFrame: The original dataframe with a new 'pred_doc_label' column.
    """
    predicted_labels = []

    for i, row in testdf.iterrows():
        # 1) Extract the text
        user_query = row["truncated_text"]

        # 2) Call the async doc_classification_rag function
        #    This returns a JSON or string response from your LLM API
        result = await doc_classification_rag(
            user_query=user_query,
            vector_db=vector_db,
            model_name=model_name,
            model_params=model_params,
            top_k=top_k
        )

        # 3) Parse the LLM’s response
        #    If 'result' is a dict with 'choices', get the text. Otherwise, convert to string.
        if isinstance(result, dict) and "choices" in result:
            classification_text = result["choices"][0].get("text", "").strip()
        else:
            classification_text = str(result).strip()

        predicted_labels.append(classification_text)

    # 4) Store predictions in a new column
    testdf["pred_doc_label"] = predicted_labels
    return testdf

# Example usage in an async context (e.g., inside an async function or event loop)
# testdf has a column 'truncated_text'
# doc_classification_rag is your async RAG function
updated_testdf = await batch_doc_classification_rag(
    testdf=testdf,
    vector_db=collection,        # your Chroma DB
    model_name="Meta-llama-3.3-7B-Instruct",
    model_params={
        "temperature": 0.2,
        "top_p": 0.95
    },
    top_k=3
)

# Now 'updated_testdf' has a new column 'pred_doc_label' with the model’s predictions
print(updated_testdf[["truncated_text", "label", "pred_doc_label"]].head())


async def batch_combined_classification_rag(testdf, vector_db, model_name, model_params, top_k=3):
    """
    Iterates over each row in 'testdf', calls combined_classification_rag asynchronously,
    and stores both doc type and first-page predictions in new columns:
      - 'pred_doc_type'
      - 'pred_is_first'
    """
    pred_doc_type = []
    pred_is_first = []

    for i, row in testdf.iterrows():
        user_query = row["truncated_text"]

        result = await combined_classification_rag(
            user_query=user_query,
            vector_db=vector_db,
            model_name=model_name,
            model_params=model_params,
            top_k=top_k
        )

        # Parse the LLM’s response
        if isinstance(result, dict) and "choices" in result:
            answer_str = result["choices"][0].get("text", "").strip()
        else:
            answer_str = str(result).strip()

        # Suppose the model returns something like "Bank Statement:True"
        if ":" in answer_str:
            doc_type, is_first = answer_str.split(":", 1)
            doc_type = doc_type.strip()
            is_first = is_first.strip()
        else:
            # If the model doesn't follow the format
            doc_type = answer_str
            is_first = "Unknown"

        pred_doc_type.append(doc_type)
        pred_is_first.append(is_first)

    testdf["pred_doc_type"] = pred_doc_type
    testdf["pred_is_first"] = pred_is_first
    return testdf


# Example usage in an async context
updated_testdf = await batch_combined_classification_rag(
    testdf=testdf,
    vector_db=collection,
    model_name="Meta-llama-3.3-7B-Instruct",
    model_params={"temperature": 0.2, "top_p": 0.95},
    top_k=3
)

print(updated_testdf[["truncated_text", "label", "is_first_pg", "pred_doc_type", "pred_is_first"]].head())
