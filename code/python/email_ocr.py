Subject: Consolidated OCR File Format (includes raw OCR text by page)

Hi Harmeet,
As discussed, here’s the proposed consolidated CSV format we’ll use for test/calibration datasets. Each row = one document. It includes: identity/lineage, OCR provenance, raw OCR text (both concatenated and per-page JSON), and optional GT/prediction columns.
I’ve attached (1) the template CSV with two example rows and (2) a data dictionary with column types and rules. A small Pandas validator is included below to sanity-check population during testing.
If this works, we’ll publish v1.0 for W-2 and Statements and keep it versioned for future changes.
Thanks!
