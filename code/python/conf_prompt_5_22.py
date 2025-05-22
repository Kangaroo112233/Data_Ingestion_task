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
Mailing Address Line 1: {row['Mailing Address Line 1']}

Instructions:
1. Check if both **First Name** and **Last Name** from the System of Record are present in the document as exact or partial **substring** matches (case-insensitive).
   - Ignore middle names completely if present in the document.
   - A partial match means the SoR name is contained within the document name (e.g., "Rob" in "Robert", "Ann" in "Annabelle").
   - If both names match, return YES.

2. If both names do not match, check whether the **Mailing Address Line 1** matches (exact or partial **substring** match).
   - Example: "123 Main St" in the document can match "123 Main Street" in the SoR.
   - If this matches, return YES.

3. If neither condition is satisfied, return NO.

If multiple names are mentioned in the document:
- Identify all names.
- Choose the **primary subject** (account holder, patient, decedent, addressee, etc.).
- Use only the primary subject's name for comparison.

Respond ONLY in the following strict JSON format:
{{ "decision": "yes" }} or {{ "decision": "no" }}

DO NOT include any explanations, reasoning, or additional text in your response.
"""
