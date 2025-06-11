
Returned-Mail Handling: Confirm by June 12 whether the regex-based “returned_mail” bypass stays upstream (and hard-codes 100% confidence) or moves post-inference.

Transaction-Table Extraction: Ship a regex-fallback on descriptions (Payment|Paid|Received), tune OCR on the last row, and make the “check first N rows for ‘–’” window configurable.

Invoice Extraction Tweaks:

Prompt-tweak + block-grab regex for multi-line “Bill To” addresses.

Extend LLaMA prompt (and add regex overrides) for tax-bill fields (taxable_amount, cgst_amount, etc.).

“Other” Bucket Samples: Request Devoteam for 20–25 raw PDFs that don’t fit known templates (foreign-language, unreadable, etc.) to train a binary “Known vs. Other” classifier.

Calibration Baseline: Extract LLaMA log-probs, compute raw ECE/MCE/Brier, and produce a reliability diagram by COB tomorrow; run temperature scaling & isotonic experiments by Friday.

Tech Stack Reminder:

OCR: Microsoft OCR

Classification: ModernBERT

Extraction: LLaMA 3.3 70B

Low-Latency Inference: Engage Triton + NVIDIA TensorRT-LLM—email HPC team for their 4–5-bullet setup summary today, then switch inference to Triton and benchmark FP16/INT8 vs. baseline by Friday.
