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
