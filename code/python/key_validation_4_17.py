Auto-Key Validation and Confirmation Model Summary

Objective:
Automate validation of documents to ensure they are ingested to the correct customer profile, preventing Non-Public Information (NPI) violations, and enabling Straight Through Processing (STP).

Validation Logic Overview:

Matching Strategy:

Validate if a document matches System of Record (SoR) data based on:

Name

Address

General Rules:

Minimum 2 out of 3 fields must match (Name, Address, or other designated index key).

No conflicting values must be present.

Matching is semantic, not exact (e.g., "Bob" is valid for "Robert").

Partial values are allowed (e.g., "123 Main Street" vs. full address).

If even one element is a mismatch, the document should be routed to HITL.

We do not want to hard-code rules into the model but explore multiple modeling strategies for best performance.

Validation Scenarios:

Scenario

Name

Address

Scenario 2

X



Scenario 3

X

X

Scenario 4



X

Sample Strategy:

DIIVO team to provide 5-10 samples per scenario by end of week.

Team may also generate altered document versions to create wider test variety.

Owen/Matt will share a detailed scenario breakout for implementation guidance.

Model Behavior:

Model receives a document and expected SoR values.

Prompt to the model: "Does this document belong to the customer based on available data? Yes/No"

The model should:

Make a judgment based on semantic similarity and context

Be conservative: Fail if uncertain, route to HITL

Output a confidence score, which will be used as a fallback trigger for manual review

Two Layers of Variability:

Value Representation Variability

Synonyms (Bob = Robert), abbreviations (St = Street), partials (123 Main St = 123 Main Street, Tampa FL)

Field Presence Variability

Sometimes only name or only address is available

Model should evaluate in totality, considering what data is available and reliable

Scenario Expectation Matrix (To be co-developed with business):

Scenario ID

Name Present

Address Present

Conflict Present

Expected Action

Notes

S2

Yes

No

No

Approve

Partial address okay

S3

Yes

Yes

No

Approve

Full match preferred

S4

No

Yes

No

HITL

Missing name field

Hybrid Prompt Strategy (Recommended):

Prompt: "Extract name and address from this document. Do they semantically match the SoR values?"

Model makes judgment based on available data

Output: Yes/No + Confidence Score
