FORM_1008_EXTRACTION_PROMPT_V2 = """
You are a data extraction (Named Entity Recognition - NER) model. For the provided
Fannie Mae Form 1008 (Mortgage Underwriting Transmittal Summary), extract the following 51 fields.
Return ONLY pretty-printed JSON (name–value pairs). If a field is missing, return "NULL".
Return every value as a string exactly as it appears in the document (keep $, %, negatives, and checkboxes).

{
  "total_borrower_income": "Total borrower income.",
  "time_stamp": "Document timestamp/date shown on the form.",
  "sales_price": "Sale price of property.",
  "representative_score": "Decision credit score.",
  "rental_income_subject_property": "Rental income for the subject property.",
  "qualifying_ratios_front_end_dti": "Front End DTI (debt-to-income).",
  "qualifying_ratios_back_end_dti": "Back End DTI (debt-to-income).",
  "property_type": "Property type (checkbox with multiple options; more than one may be selected).",
  "property_address": "Property address (street, city, state, ZIP).",
  "occupancy_type": "Occupancy status (checkbox with multiple options).",
  "note_rate": "Interest rate (note rate).",
  "months_reserves": "Number of months reserves borrower has.",
  "net_rental_income_other_property": "Net rental income from other property.",
  "ltv": "Loan-to-Value ratio.",
  "loan_type": "Loan type (checkbox; multiple options possible).",
  "loan_amount": "Total commitment / loan amount.",
  "loan_term": "Loan term (e.g., months or years, print exactly as shown).",
  "loan_purpose": "Loan purpose (checkbox; multiple options possible).",

  "loan_number": "Loan application number.",
  "lien_position": "Lien position (only checkbox options for first or second mortgage).",
  "hcltv": "High credit LTV / HTLTV with junior liens (home equity).",

  "first_mortgage_pi": "First mortgage principal & interest payment amount.",
  "subordinate_liens_pi": "Subordinate liens principal & interest payment amount(s).",
  "homeowner_insurance": "Homeowner insurance payment amount.",
  "property_tax": "Property tax payment amount.",
  "mortgage_insurance": "Mortgage insurance payment amount.",
  "association_dues_hoa": "Association/Project dues (HOA) payment amount.",
  "total_monthly_payment": "Total monthly payment amount.",
  "escrow": "Escrow checkbox – 'Yes' or 'No'.",

  "document_date": "Document date (if shown separately from timestamp).",
  "cltv": "Combined Loan-to-Value.",

  "borrower_income_borrower2": "Borrower 2 income.",
  "borrower_income_borrower1": "Borrower 1 income.",
  "borrower_income_borrower3": "Borrower 3 income.",
  "borrower_income_borrower4": "Borrower 4 income.",
  "borrower_income_others": "Borrower 5+ and other income.",
  "borrower_self_employed": "Borrower self-employed checkbox.",

  "other_monthly_payments": "Other monthly payment amount used in qualifying.",
  "funds_to_close_verified": "Borrower funds to close – verified assets amount.",
  "funds_to_close_required": "Borrower funds to close – required amount.",

  "risk_assessment": "Risk assessment checkbox – underwriting type 'Manual Undw / AUS' with subtype if present (DU, LPA, Other).",
  "aus_recommendation": "AUS recommendation/result.",

  "appraised_value": "Appraised value of collateral property.",
  "appraisal_type": "Appraisal type (checkbox; multiple options possible).",
  "appraisal_form_number": "Appraisal form number / appraisal vendor code.",
  "subordinate_financing": "Amount of subordinate financing.",
  "amortization_type": "Amortization type (checkbox; multiple options possible).",

  "project_class_fhlmc": "Project classification checkbox for Freddie Mac (multiple values may be selected).",
  "project_class_fnma": "Project classification checkbox for Fannie Mae (multiple values may be selected).",
  "fnma_project_id": "Project Manager project ID# for Fannie Mae.",
  "project_name": "Project name for Fannie Mae."
}

Extraction Rules:
1) Extract exactly these 51 fields. If any is not present, output "NULL" for that key.
2) Output must be JSON only — no prose, no code fences.
3) Preserve symbols and formatting ($, %, parentheses, minus/negative prefixes, and checkbox labels).
4) For checkboxes with multiple options, return the exact selected label(s) as a single string; if multiple are checked, join using a semicolon and a space (e.g., "Primary Residence; Investment").
5) Dates: return exactly as shown (do not reformat).
6) Numeric amounts and rates: keep dollar signs, decimals, commas, and negative prefixes if present.
Stop after printing the JSON.
"""
