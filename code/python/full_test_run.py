import json

results = []

for idx, document in tqdm(df.iterrows()):
    text = document['text']

    output = await prompt_model(
        query=text,
        system_prompt=system_prompt,
        user_role="document",
        model_name=model_name,
        temperature=1e-3,
        top_p=0.95,
        logprobs=True,
        print_output=False,
    )

    # Extract raw response
    raw_result = output['choices'][0]['message']['content'].strip()

    # Attempt to clean JSON formatting issues
    try:
        # Fix incomplete JSON structure by extracting valid JSON
        json_start = raw_result.find("{")
        json_end = raw_result.rfind("}") + 1
        cleaned_json = raw_result[json_start:json_end]

        # Convert string to valid JSON object
        result = json.loads(cleaned_json)

    except json.JSONDecodeError:
        print(f"JSON Error at index {idx}: Trying alternate fix...")
        
        # Alternative JSON Fix: Removing unwanted trailing characters
        try:
            cleaned_json = raw_result.split("```json")[-1].split("```")[0].strip()
            result = json.loads(cleaned_json)
        except Exception as e:
            print(f"Final JSON Error at index {idx}: {e}")
            result = {}

    results.append(result)

# Convert results into a DataFrame
results_df = pd.DataFrame(results)
tools.display_dataframe_to_user(name="Final Extracted Data", dataframe=results_df)
