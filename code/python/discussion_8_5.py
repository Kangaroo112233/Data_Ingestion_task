1. Vendor Name Extraction
Issue: Model often returns the issuing institution (e.g. “Bank of America”) instead of the payment-coupon’s vendor line (e.g. “Business Card”).

Decision: Ground truth for the vendor name should match the coupon’s “Vendor Name” line. If the coupon shows “Business Card,” that’s what the model should learn to extract.

2. Apple Card Statements
Issue: Apple Card PDFs don’t include a payment coupon, so the model pulls institution names and addresses from elsewhere, yielding “iffy” results.

Decision: Exclude Apple Card statements from end-to-end testing (i.e. remove those files from the answer key), since they’ll never be sent through production.

3. Transaction vs. Invoice Paths
Clarification: Even for statements without coupons (Apple Card, Robin Hood), transaction data still needs to go through the model’s transaction/invoice path. Keep those fields enabled.

4. Synchrony/Amazon Outliers
Issue: Some Synchrony-issued Amazon statements came in scanned without coupons, causing the model to extract PO Boxes from disclaimer sections.

Decision: Treat those as rare “bad scans” — for grading, expect nulls for coupon fields when the coupon is genuinely missing.

5. Robin Hood Credit Card Cases
Issue: Robin Hood statements lack a coupon but include an address that isn’t a valid payment address.

Action Item:

Confirm whether Robin Hood payments are ever made (i.e. whether those records matter).

If not, treat them like Apple Card (exclude from coupon-field testing), but keep transaction fields active.

6. Address Formatting (Commas)
Observation: Post-processing strips commas inserted by the model at line-break boundaries (e.g. “P.O. Box 1423, Charlotte, North Carolina”).

Decision: It’s fine to leave those commas in the output—downstream parsers can handle them as separators.

7. Total Charges vs. Total Due
Question: Both fields often match, so is one redundant?

Clarification: They can differ when partial payments occur (e.g. charges $10,000 vs. due $7,000).

Decision: Keep both as distinct fields.

Next Steps / Action Items

Update the ground-truth labels to use the coupon’s vendor name (e.g. “Business Card”).

Remove Apple Card files from the coupon-field test set.

Confirm Robin Hood payment inclusion; if confirmed unused, exclude also.

Ensure the test harness treats missing coupons as nulls (not as mis-extractions).

Leave commas in extracted addresses.

Retain both “Total Charges” and “Total Due” fields in evaluation.

Let me know if you’d like any of these turned into formal meeting minutes or emailed to the team!









Ask ChatGPT

