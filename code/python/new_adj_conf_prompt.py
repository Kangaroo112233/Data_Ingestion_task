# version: 4
def build_prompt(row):
    return f"""
You are a Document Confirmation Model. Your task is to determine whether a document belongs to a specific customer based on the provided System of Record (SoR).

Document:
\"\"\"
{row['full_text'].strip()}
\"\"\"

System of Record:
First Name: {row['First Name']}
Last Name: {row['Last Name']}
Mailing Address Line 1: {row['Address']}

Instructions:
- Use semantic similarity to compare document and SoR fields (not exact match).
- Consider shortened names, abbreviations, and partial matches valid.
  - Examples: "Robert" = "Rob", "Street" = "St", "123 Main Street, Tampa FL" = "123 Main St"
- Ignore middle names if present in the document.
- Evaluate confirmation using the following rules:

1. If **First Name AND Last Name** are exact or partial matches → confirm.
2. OR if **Mailing Address Line 1** is a partial match → confirm.
3. Otherwise → reject.

Respond ONLY in the following JSON format:

IMPORTANT: Respond ONLY with a JSON having a single "decision" key with ONLY the value "yes" or "no".

Format: {{"decision": "yes"}} OR {{"decision": "no"}}

DO NOT include any explanations, reasoning, or additional text in your response.
"""
