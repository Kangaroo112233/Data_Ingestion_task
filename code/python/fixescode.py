# Evaluation: Compare ground-truth metadata to the predicted metadata from search results.

pred_label_l, pred_first_pg_l, pred_pg_num_l, pred_score_l = [], [], [], []

for idx, results in enumerate(lolo_results):
    print(f"Processing document#: {idx}")

    # Ensure results[0] is a dictionary
    if isinstance(results, list) and len(results) > 0:
        first_element = results[0]
        if isinstance(first_element, list) and len(first_element) > 0:
            result_dict = first_element[0]  # Extract dictionary from nested list
        elif isinstance(first_element, dict):
            result_dict = first_element  # Use it directly
        else:
            print(f"Skipping index {idx}, unexpected format:", first_element)
            continue
    elif isinstance(results, dict):
        result_dict = results  # Use results directly if it's already a dictionary
    else:
        print(f"Skipping index {idx}, unexpected format:", results)
        continue

    # Extract distance values safely
    distances = result_dict.get('distances', [[0]])[0]
    if isinstance(distances, list):
        cos_sim = 1 - max(0, *distances)  # Compute similarity score
    else:
        print(f"Skipping index {idx}, invalid distance format:", distances)
        continue

    # Extract metadata values safely
    if 'metadatas' in result_dict and isinstance(result_dict['metadatas'], list) and len(result_dict['metadatas']) > 0:
        metadata_list = result_dict['metadatas']
        
        # Ensure metadata_list contains dictionaries
        metadata = metadata_list[0] if isinstance(metadata_list[0], dict) else {}
        
        pred_label_l.append(metadata.get('label', 'Unknown'))
        pred_first_pg_l.append(metadata.get('first_pg', False))
        pred_pg_num_l.append(metadata.get('pg_num', -1))
    else:
        print(f"Skipping index {idx}, metadata not found")
        continue

    pred_score_l.append(cos_sim)

# Extract ground-truth metadata
label_l, first_page_l = [], []

for idx, results in enumerate(srch_lolo_metadata):
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
        label_l.append(results[0].get('label', 'Unknown'))
        first_page_l.append(results[0].get('first_pg', False))
    else:
        print(f"Skipping index {idx}, unexpected format in ground truth metadata:", results)

print("Evaluation Completed")



def print_search_results(lolo_results, k=1):
    """
    Print search results from FAISS vector database in a clear, formatted way.
    
    Args:
        lolo_results: List of lists of results from FAISS search
        k: Number of nearest neighbors retrieved per query (default=1)
    """
    print("\n=== Search Results Summary ===")
    print(f"Total search documents: {len(lolo_results)}")
    print(f"Results per search (k): {k}")
    print("=============================\n")
    
    for doc_idx, doc_results in enumerate(lolo_results):
        print(f"\nDocument #{doc_idx + 1}")
        print("─" * 40)
        
        # Handle each chunk's results
        for chunk_idx, chunk_result in enumerate(doc_results):
            print(f"\nChunk {chunk_idx + 1}:")
            
            # Extract the nested results structure
            # FAISS results are structured as: [{'ids': [[...]], 'distances': [[...]], etc}]
            result_dict = chunk_result[0]  # First (and only) result dictionary
            
            # Get distances and compute similarity scores
            distances = result_dict['distances'][0]  # First list of distances
            similarities = [1 - max(0, dist) for dist in distances]
            
            # Print each matching result
            for i in range(len(distances)):
                print("\n  Match Details:")
                print(f"  • Similarity Score: {similarities[i]:.2%}")
                
                # Print ID if available
                if 'ids' in result_dict and result_dict['ids'][0]:
                    print(f"  • ID: {result_dict['ids'][0][i]}")
                
                # Print metadata if available
                if 'metadatas' in result_dict and result_dict['metadatas'][0]:
                    metadata = result_dict['metadatas'][0][i]
                    print("  • Metadata:")
                    for key, value in metadata.items():
                        print(f"    - {key}: {value}")
                
                # Print document content if available
                if 'documents' in result_dict and result_dict['documents'][0]:
                    doc_content = result_dict['documents'][0][i]
                    print("  • Document Content:")
                    # Truncate long documents for display
                    max_chars = 200
                    if len(doc_content) > max_chars:
                        doc_content = doc_content[:max_chars] + "..."
                    print(f"    {doc_content}")
            
            print("\n" + "·" * 30)  # Separator between chunks
        
        print("─" * 40)  # Separator between documents


 # Safely access metadata
            metadatas = result.get('metadatas', [[]])[0]
            if metadatas and isinstance(metadatas, list) and len(metadatas) > 0:
                metadata = metadatas[0]
            else:
                metadata = metadatas if metadatas else {}
                
            # Append predictions with safe gets
            pred_label_l.append(metadata.get('label', 'unknown'))
            pred_first_pg_l.append(metadata.get('first_pg', -1))
            pred_pg_num_l.append(metadata.get('pg_num', -1))
            pred_score_l.append(sim_score)
            
        except Exception as e:
            print(f"Error processing result {idx}: {str(e)}")
            print(f"Result content: {results}")
            # Append default values on error
            pred_label_l.append('error')
            pred_first_pg_l.append(-1)
            pred_pg_num_l.append(-1)
            pred_score_l.append(0)
    
    # Initialize lists for ground truth
    label_l = []
    first_page_l = []
    
    # Process ground truth from metadata
    for idx, metadata in enumerate(srch_lolo_metadata):
        try:
            # Safely access ground truth metadata
            if metadata and isinstance(metadata, list) and len(metadata) > 0:
                label_l.append(metadata[0].get('label', 'unknown'))
                first_page_l.append(metadata[0].get('first_pg', -1))
            else:
                label_l.append('unknown')
                first_page_l.append(-1)
        except Exception as e:
            print(f"Error processing metadata {idx}: {str(e)}")
            label_l.append('error')
            first_page_l.append(-1)
