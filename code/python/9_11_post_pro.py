

Thanks for your feedback. Yes, this format is intended to serve as the Ground Truth Excel/CSV that will also align with the Evaluation Excel requirements.

To clarify what will be included in the consolidated file (1 row = 1 document):

File metadata: file name, path, checksum, page count, ingest timestamp

Raw OCR output:

ocr_text → concatenated text with ---PAGE_BREAK--- delimiters

ocr_pages_json → JSON list of page-level text strings

Ground truth columns: populated based on the annotation spec

Optional prediction columns: to be added when model outputs are integrated

Versioning/traceability: run IDs, rules version, calibration/test split flag

This way, both the GT Excel and the Evaluation Excel will reconcile against the same schema, ensuring consistency across test and calibration datasets.
