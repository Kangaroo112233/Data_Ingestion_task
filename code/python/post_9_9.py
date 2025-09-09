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
