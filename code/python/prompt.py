Task:
Extract structured fields from the To Address section of an envelope or document. Ensure no information is extracted from the From Address or any other parts of the document.

Context:
These documents contain addresses that may belong to individuals, businesses, or departments. The focus is to extract only the recipient information.

{
  "Name": "Extract the full name of the recipient (can be an individual or company name). Prioritize company names when available.",
  "Street Address or PO Box": "Extract the street address if available. If not, extract the PO Box.",
  "City": "Extract the city name.",
  "State": "Extract the two-letter state code.",
  "ZIP Code": "Extract the ZIP code.",
  "Mail Code": "Extract the mail code if it exists (often labeled as 'Mail Code' or similar). If not present, return 'NULL'.",
  "Business Group Name": "Extract if available (commonly found after 'ATTN:' or similar keywords). If not present, return 'NULL'."
}

Additional Extraction Rules:
Extract Only from the To Address:

Ignore any information from the sender (From Address).
If multiple addresses exist, extract the primary recipient.
Handling Variability in Address Format:

If the recipient name contains multiple names (e.g., two individuals), include both.
If only a company name is present, use that as the Name.
If a Mail Code is missing, return "NULL".
Ensure Output is in JSON Format:

Return the extracted data strictly as a JSON object.
Each field should appear only once.
If a field is not present in the document, return "NULL" instead of skipping it.
