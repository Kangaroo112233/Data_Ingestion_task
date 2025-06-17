Below are three new sub-sections you can drop into your trimmed-down Section 10.1 (immediately after the Performance Report and before Calibration), to address the MU and DI reviewer comments. Feel free to tweak figure/table numbering as needed.

10.1.7 Validation-Confirmation Use Case & Workflow
The Validation-Confirmation step is the final gating operation in our end-to-end pipeline. It runs after document classification and field extraction, and before ingesting results into downstream systems. If the VC model returns { "decision": "no" }, the document is routed to a manual indexing queue for human review; if { "decision": "yes" }, processing continues automatically.

pgsql
Copy
Edit
┌───────────────┐    ┌───────────────┐    ┌─────────────────────────┐    ┌──────────────┐
│ Document      │ →  │ Classification│ →  │ Data Extraction        │ →  │ Validation   │
│ Ingestion     │    │ (type-label)  │    │ (field values)         │    │ Confirmation │ 
└───────────────┘    └───────────────┘    └─────────────────────────┘    └──────┬───────┘
                                                                          Yes │ No
                                                                              ▼ 
                                                                          Manual
                                                                          Indexer
Independent Step: VC does not feed back into upstream extraction; it simply confirms ownership.

Dependencies:

Classification must assign the correct document type (so we know which SoR fields to compare).

Extraction must produce the raw text (full document) and SoR metadata.

10.1.8 Sample-Size Justification
We used 172 held-out documents to estimate zero-shot VC performance. At an observed accuracy of 0.98, the 95 % confidence-interval half-width (margin of error) is:

ME
  
=
  
𝑧
0.975
 
𝑝
(
1
−
𝑝
)
𝑛
  
=
  
1.96
×
0.98
×
0.02
172
≈
0.03
(
±
3
%
)
ME=z 
0.975
​
  
n
p(1−p)
​
 
​
 =1.96× 
172
0.98×0.02
​
 
​
 ≈0.03(±3%)
Thus we can assert, with 95 % confidence, that true accuracy lies in [95 %, 100 %]. This margin is acceptable for production gating, and we will maintain it in ongoing monitoring by re-sampling at least 172 documents per quarter.

10.1.9 Stratified Performance Metrics
We report stratification both by decision class (belongs/doesn’t belong) and, going forward, by document type.

Decision Class	Precision	Recall	F1-Score	Support
belongs (yes)	0.95	1.00	0.98	81
doesn’t belong (no)	1.00	0.96	0.98	91

Belongs: the model correctly confirmed 100 % of true-yes cases (no false negatives).

Doesn’t belong: the model correctly rejected 96 % of true-no cases.

Next steps: in our ICM dashboard we will also break these metrics out by incoming document type (e.g. Income, Statement, W2, Death Cert, Signature Card), so we can detect drift per segment.

  ┌───────────────┐    ┌───────────────┐    ┌─────────────────────────┐    ┌──────────────┐
│ Document      │ →  │ Classification│ →  │ Data Extraction        │ →  │ Validation   │
│ Ingestion     │    │ (type-label)  │    │ (field values)         │    │ Confirmation │ 
└───────────────┘    └───────────────┘    └─────────────────────────┘    └──────┬───────┘
                                                                          Yes │ No
                                                                              ▼ 
                                                                          Manual
                                                                          Indexer




  Validation Confirmation Model

Purpose: As the final step in our ESO pipeline, the validation confirmation model ensures that the extracted reference-number (or “key”) from a given document truly matches the authoritative record in our System of Record (SOR).

Inputs:

Split classification output (document type)

Reference-number extraction (from data-extraction model)

Process:

Query the SOR using the extracted reference-number

Compare the SOR-returned canonical fields (e.g. name, address) against the OCR-extracted values

Compute a per-field confidence score and aggregate into an overall “validation” confidence

Outputs:

Validation = Yes (match above threshold)

Validation = No (mismatch or low confidence)

STP Yes/No Decision
Once validation completes, we branch on the boolean outcome:

Validation Result	Action
Yes	• → Straight-Through Processing (STP)
– Automatically route the document and its confirmed metadata into the downstream ingestion/indexing pipeline.
– Mark record as STP-eligible in the audit log.
No	• → Manual Review
– Flag the document for a human-in-the-loop exception workflow.
– Attach discrepancy details and confidence scores to the review ticket.

ME = z₀.₉₇₅ × √[p(1−p)/n]
   = 1.96 × √[(0.98 × 0.02)/172]
   ≈ 0.03 (±3 %)

4. Validation Confirmation Use Case (MU)
Comment: “It is unclear if this step occurs independently of split classification & extraction or if it must follow in sequence. Please clarify dependencies in the diagram.”
Our Response:

The Validation-Confirmation (VC) step is always executed immediately after Split Classification and Data Extraction as the final check in our ESO pipeline. We have updated Figure 2 (page 9) to show the sequential dependency—Split Classification ➔ Data Extraction ➔ Validation Confirmation—and added a “Yes/No” branch from VC to either auto‐indexing (STP) or human review.

7.1 Sample Size & Stratification (PT)
Comment: “Please provide evidence that N = 172 for validation is sufficient, including margin of error, and plan for ongoing stratified monitoring.”
Our Response:

We used 172 held-out production documents to estimate zero-shot VC performance. At an observed accuracy of 0.98, the 95 % confidence-interval half-width (margin of error) is:

ME
=
𝑧
0.975
 
𝑝
(
1
−
𝑝
)
𝑛
=
1.96
×
0.98
×
0.02
172
≈
0.03
(
±
3
%
)
ME=z 
0.975
​
  
n
p(1−p)
​
 
​
 =1.96× 
172
0.98×0.02
​
 
​
 ≈0.03(±3%)
Thus, with 95 % confidence the true accuracy lies in [95 %, 100 %], which meets our STP-gating threshold. Going forward, we will re-sample at least 172 documents per quarter and report stratified metrics by “belongs/doesn’t belong” and by document type to satisfy ICM tracking requirements.

10.1 Validation Confirmation Output Flags (Doc)
Comment: “Add the ‘Yes/No’ validation outputs and downstream actions to the model workflow.”
Our Response:

We have inserted a new table in Section 10.1 that maps the VC decision to its action:

Validation Result	Action
Yes	• Straight-Through Processing (STP): auto-route confirmed documents into downstream ingestion; mark record as STP-eligible in audit logs.
No	• Manual Review: flag for human-in-the-loop exception workflow; attach per-field discrepancy details and confidence scores to the review ticket.
