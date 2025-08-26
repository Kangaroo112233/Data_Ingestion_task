import pandas as pd

# Load dataframe
df = pd.read_excel("your_file.xlsx")

# Work only on rows where field == "total_credits"
mask = df["field"] == "total_credits"

# Step 1: Clean ground_truth → remove parentheses, add negative sign
df.loc[mask, "ground_truth"] = (
    df.loc[mask, "ground_truth"]
    .str.replace(r"[\(\)]", "", regex=True)   # remove parentheses
    .str.replace(",", "", regex=True)         # remove commas
    .apply(lambda x: "-" + x if not x.strip().startswith("-") else x)
)

# Step 2: Update accuracy only for total_credits
df.loc[mask, "accuracy"] = (
    df.loc[mask, "predicted_value"].astype(str).str.strip()
    == df.loc[mask, "ground_truth"].astype(str).str.strip()
).astype(int)

# Save back if needed
df.to_excel("updated_file.xlsx", index=False)




# 1) Normalize column names to avoid hidden spaces/case issues
df_9.columns = df_9.columns.str.strip().str.lower()

# 2) Work only on total_credits rows
mask = df_9["field"].astype(str).str.strip().eq("total_credits")

# 3) Clean ground_truth: remove $ , () → add minus if it had parentheses
gt = df_9.loc[mask, "ground_truth"].astype(str).str.strip()
had_paren = gt.str.match(r'^\(.*\)$')   # remember which were in ()
gt_clean = (
    gt.str.replace(r'[\$,()]', '', regex=True)  # drop $, commas, parentheses
      .str.replace(',', '', regex=True)
)
gt_clean = gt_clean.where(~had_paren, "-" + gt_clean)  # add '-' if it had ()

df_9.loc[mask, "ground_truth"] = gt_clean

# 4) Normalize predicted_value the same way
pred_clean = (
    df_9.loc[mask, "predicted_value"].astype(str).str.strip()
       .str.replace(r'[\$,]', '', regex=True)
)

# 5) Update accuracy = 1 when they match (string compare after normalization)
df_9.loc[mask, "accuracy"] = (pred_clean == df_9.loc[mask, "ground_truth"].astype(str).str.strip()).astype(int)
