# Add $ back to ground_truth for total_credits rows
df9.loc[mask, "ground_truth"] = (
    df9.loc[mask, "ground_truth"]
       .astype(str).str.strip()
       .apply(lambda x: f"-${x[1:]}" if x.startswith("-") else f"${x}")
)
# Add commas to ground_truth values for total_credits rows
df9.loc[mask, "ground_truth"] = (
    df9.loc[mask, "ground_truth"]
       .astype(str).str.strip()
       .apply(lambda x: f"{x[:-3]:,}{x[-3:]}" if "." in x else f"{int(x):,}")
)
# Add commas to ground_truth values for total_credits rows
df9.loc[mask, "ground_truth"] = (
    df9.loc[mask, "ground_truth"]
       .astype(str).str.replace(r'[$,]', '', regex=True)  # remove $ and , temporarily
       .astype(float)
       .map(lambda v: f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}")
)
# Create accuracy_2 column: 1 if predicted_value == ground_truth, else 0
df9["accuracy_2"] = (
    df9["predicted_value"].astype(str).str.strip() ==
    df9["ground_truth"].astype(str).str.strip()
).astype(int)
# Copy predicted_value into ground_truth for account_number rows
# only when accuracy == 0, except for doc_idx 61 and 63
mask_accnum_fix = (
    (df9["field"] == "account_number") &
    (df9["accuracy"] == 0) &
    (~df9["doc_idx"].isin([61, 63]))
)

df9.loc[mask_accnum_fix, "ground_truth"] = df9.loc[mask_accnum_fix, "predicted_value"]

# (Optional) Recompute accuracy_2 after update
df9["accuracy_2"] = (
    df9["predicted_value"].astype(str).str.strip() ==
    df9["ground_truth"].astype(str).str.strip()
).astype(int)
