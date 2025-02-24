import asyncio

# Create lists to store results
doc_classification_results = []
first_page_results = []
combined_results = []

async def process_test_documents():
    tasks = []

    for _, row in testdf.iterrows():
        test_doc_text = row["truncated_text"]

        # Document Classification
        tasks.append(
            doc_classification_rag(
                user_query=test_doc_text,
                vector_db=vector_db,  # Your Chroma DB collection
                model_name="Meta-Llama-3.3-70B-Instruct",
                model_params={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "logprobs": True,
                    "prompt_type": "instruction"
                },
                top_k=3
            )
        )

        # First-Page Classification
        tasks.append(
            first_page_classification_rag(
                user_query=test_doc_text,
                vector_db=vector_db,  # Your Chroma DB collection
                model_name="Meta-Llama-3.3-70B-Instruct",
                model_params={
                    "temperature": 0.2,
                    "top_p": 0.95
                },
                top_k=3
            )
        )

        # Combined Classification
        tasks.append(
            combined_classification_rag(
                user_query=test_doc_text,
                vector_db=vector_db,  # Your Chroma DB collection
                model_name="Meta-Llama-3.3-70B-Instruct",
                model_params={
                    "temperature": 0.2,
                    "top_p": 0.95
                },
                top_k=3
            )
        )

    # Run all tasks asynchronously
    results = await asyncio.gather(*tasks)

    # Store results
    for i in range(0, len(results), 3):
        doc_classification_results.append(results[i])
        first_page_results.append(results[i + 1])
        combined_results.append(results[i + 2])

# Run the async function
await process_test_documents()

# Print the first few results for verification
print("Sample Document Classification Results:", doc_classification_results[:3])
print("Sample First-Page Classification Results:", first_page_results[:3])
print("Sample Combined Classification Results:", combined_results[:3])
