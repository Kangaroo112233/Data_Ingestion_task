def build_prompt(row):
    return f"""
You are a helpful AI assistant. Your task is to determine whether a document belongs to a specific customer based on the provided System of Record (SoR).

Document:
\"\"\"
{row['doc_text'].strip()}
\"\"\"

System of Record:
Name: {row['sor_name']}
Address: {row['sor_address']}

Check whether the document belongs to this customer using semantic similarity (not exact matching). Consider shortened names and partial addresses as valid. If any conflict is found, reject.

Respond ONLY in the following JSON format:
{{
  "decision": "yes" or "no",
  "confidence": float between 0 and 1,
  "explanation": "short justification"
}}
"""
