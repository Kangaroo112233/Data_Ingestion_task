import asyncio
import httpx

async def retry_api_call(client, url, headers, request_body, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = await client.post(url, headers=headers, json=request_body)
            response.raise_for_status()  # Raise error if response status is not 2xx
            return response
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code} - Retrying {attempt + 1}/{max_retries}")
        except httpx.RequestError as e:
            print(f"Request Error: {e} - Retrying {attempt + 1}/{max_retries}")
        await asyncio.sleep(2)  # Delay before retry
    return None  # Return None after all retries fail

async def prompt_model(query, system_prompt, model_name, model_params, vector_db, top_k):
    request_body = {
        "query": query,
        "system_prompt": system_prompt,
        "model_name": model_name,
        "model_params": model_params,
        "top_k": top_k
    }

    async with httpx.AsyncClient(timeout=600.0, verify=False) as client:
        response = await retry_api_call(client, phoenix_genai_service_url, headers, request_body)
        if response is None:
            print("API request failed after retries")
            return None
        return response.json()


async def process_test_documents(batch_size=5):
    tasks = []

    for idx, row in testdf.iterrows():
        test_doc_text = row["truncated_text"]

        # Create tasks for each document
        tasks.append(doc_classification_rag(test_doc_text, vector_db, model_name, model_params, top_k=3))
        tasks.append(first_page_classification_rag(test_doc_text, vector_db, model_name, model_params, top_k=3))
        tasks.append(combined_classification_rag(test_doc_text, vector_db, model_name, model_params, top_k=3))

        # Run tasks in smaller batches instead of all at once
        if len(tasks) >= batch_size:
            results = await asyncio.gather(*tasks)
            process_results(results)  # Function to handle the results
            tasks = []  # Clear tasks before starting the next batch

    # Run remaining tasks if any
    if tasks:
        results = await asyncio.gather(*tasks)
        process_results(results)


#try only when it is safe
async with httpx.AsyncClient(timeout=600.0, verify=False) as client:


# print debug logs

import traceback

async def prompt_model(query, system_prompt, model_name, model_params, vector_db, top_k):
    request_body = {
        "query": query,
        "system_prompt": system_prompt,
        "model_name": model_name,
        "model_params": model_params,
        "top_k": top_k
    }

    async with httpx.AsyncClient(timeout=600.0, verify=False) as client:
        try:
            response = await client.post(phoenix_genai_service_url, headers=headers, json=request_body)
            response.raise_for_status()  # Raise error if response status is not 2xx
            return response.json()
        except Exception as e:
            print("Error in API call:")
            traceback.print_exc()
            return None

