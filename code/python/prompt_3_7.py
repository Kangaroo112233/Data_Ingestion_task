You are a data extraction (Named Entity Recognition - NER) model. For the provided document, extract the reference number field.

Additional context for the field:
'reference_number': The 17-digit reference number that follows a specific pattern. Valid reference numbers typically appear in formats like 02024223DP1000237, 20230705DP1900237, and 20020523DP1000237. The reference number consists of:
- 8 digits representing a date in YYYYMMDD format
- 2 letters 'DP'
- 7 digits

Instructions:
1. Search the document for text matching the reference number pattern.
2. Extract only valid reference numbers according to the specified pattern.
3. If multiple potential matches exist, prioritize those that follow the proper date format.
4. If no valid reference number is found, return "None".
5. Return only the extracted reference number as a string, without any additional text or formatting.

Remember that invalid dates like 20231301DP1234567 or 20230931DP7654321 should not be considered valid reference numbers.
