for idx, document in tqdm(df.iterrows()):
    try:
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
        
        try:
            # Try to extract result from expected structure
            result = output['choices'][0]['message']['content'].replace('', '').strip()
            
            # Temp fix
            result = '{' + result.split('{', 1)[-1]
            if len(result.rsplit('}', 1)) == 1:
                # print('Attempting to fix JSON...')
                result += '}'
            else:
                result = result.rsplit('}', 1)[0] + '}'
                
            result = json.loads(result)
            results.append(result)
            
        except (TypeError, KeyError, IndexError, json.JSONDecodeError) as e:
            # Handle case where output doesn't have expected structure
            print(f"Error processing record {idx}: {e}")
            # Create empty result with None values
            empty_result = {
                "Bill Date": None,
                "Due Date": None,
                "Bill to Name": None,
                "Bill to Address": None,
                "Vendor Name": None,
                "Vendor Address": None,
                "Account Number": None,
                "Total Due": None,
                "Invoice Number": None
            }
            results.append(empty_result)
            
    except Exception as e:
        # Catch-all for any other errors
        print(f"Unexpected error on record {idx}: {e}")
        empty_result = {
            "Bill Date": None,
            "Due Date": None,
            "Bill to Name": None,
            "Bill to Address": None,
            "Vendor Name": None,
            "Vendor Address": None,
            "Account Number": None,
            "Total Due": None,
            "Invoice Number": None
        }
        results.append(empty_result)

results_df = pd.DataFrame(results)
results_df = pd.concat([df, results_df], axis=1)
results_df
