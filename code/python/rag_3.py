from model_service_api_util import prompt_model

async def doc_classification_rag(
    user_query: str,
    vector_db,
    model_name: str,
    model_params: dict,
    top_k: int = 3
) -> str:
    """
    1) Retrieves context from Chroma DB.
    2) Constructs final prompt with DOC_CLASSIFIER_SYSTEM_PROMPT + context.
    3) Calls the remote model via prompt_model(...) to classify doc type.
    """
    # Step 1: Retrieve relevant chunks
    context = rag_retrieve(user_query, vector_db, k=top_k)

    # Step 2: Build the user portion of the prompt
    # In this approach, 'system_prompt' and 'query' are separate parameters to prompt_model(...).
    # The system prompt is the classification instructions, and the 'query' is the combined context + user question.
    final_user_query = f"Document Text:\n{context}\n\nQuestion: {user_query}\nAnswer:"

    # Step 3: Call the model service API
    # 'prompt_model' is presumably asynchronous, so we use 'await'.
    result = await prompt_model(
        query=final_user_query,
        system_prompt=DOC_CLASSIFIER_SYSTEM_PROMPT,
        model_name=model_name,
        model_params=model_params,
        top_k=top_k
    )
    return result


result = await doc_classification_rag(
    user_query="Some text from the page that might be a bank statement or paystub.",
    vector_db=collection,  # your Chroma DB collection
    model_name="Meta-llama-3.3-7B-Instruct",
    model_params={
        "temperature": 0.2,
        "top_p": 0.95,
        "logprobs": True,
        "prompt_type": "instruction"
    },
    top_k=3
)
print("Document Classification Result:", result)


async def first_page_classification_rag(
    user_query: str,
    vector_db,
    model_name: str,
    model_params: dict,
    top_k: int = 3
) -> str:
    """
    1) Retrieves context from Chroma DB.
    2) Constructs final prompt with SPLIT_CLASSIFIER_SYSTEM_PROMPT + context.
    3) Calls the remote model to classify if it's first page (True/False).
    """
    context = rag_retrieve(user_query, vector_db, k=top_k)
    final_user_query = f"Page Text:\n{context}\n\nQuestion: {user_query}\nAnswer:"

    result = await prompt_model(
        query=final_user_query,
        system_prompt=SPLIT_CLASSIFIER_SYSTEM_PROMPT,
        model_name=model_name,
        model_params=model_params,
        top_k=top_k
    )
    return result


result = await first_page_classification_rag(
    user_query="Here is the first page of something with a header and summary.",
    vector_db=collection,
    model_name="Meta-llama-3.3-7B-Instruct",
    model_params={
        "temperature": 0.2,
        "top_p": 0.95
    },
    top_k=3
)
print("First-Page Classification Result:", result)


async def combined_classification_rag(
    user_query: str,
    vector_db,
    model_name: str,
    model_params: dict,
    top_k: int = 3
) -> str:
    """
    Single call to do both doc-type classification and first-page detection.
    1) Retrieves context from Chroma DB.
    2) Uses CLASSIFIER_SYSTEM_PROMPT + context + user_query.
    3) Calls the remote model for a combined classification result.
    """
    context = rag_retrieve(user_query, vector_db, k=top_k)
    final_user_query = f"Document Text:\n{context}\n\nQuestion: {user_query}\nAnswer:"

    result = await prompt_model(
        query=final_user_query,
        system_prompt=CLASSIFIER_SYSTEM_PROMPT,
        model_name=model_name,
        model_params=model_params,
        top_k=top_k
    )
    return result


result = await combined_classification_rag(
    user_query="This document might be a paystub, and it's page 1 with the employee's info.",
    vector_db=collection,
    model_name="Meta-llama-3.3-7B-Instruct",
    model_params={
        "temperature": 0.2,
        "top_p": 0.95
    },
    top_k=3
)
print("Combined Classification Result:", result)
