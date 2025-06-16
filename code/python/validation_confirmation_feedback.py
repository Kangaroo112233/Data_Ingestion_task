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

  
