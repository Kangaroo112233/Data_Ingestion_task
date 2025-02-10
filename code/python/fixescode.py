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
