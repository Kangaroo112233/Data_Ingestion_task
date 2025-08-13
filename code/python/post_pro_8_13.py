# rows for the "total_credits" field
m = df5['field'].eq('total_credits')

# 1) remove the leading "-" from predicted_value (keep $/commas)
df5.loc[m, 'predicted_value'] = (
    df5.loc[m, 'predicted_value'].astype(str).str.replace(r'^\s*-\s*', '', regex=True)
)

# 2) remove parentheses from ground_truth
df5.loc[m, 'ground_truth'] = (
    df5.loc[m, 'ground_truth'].astype(str).str.replace(r'[()]', '', regex=True)
)


m = df5['field'].eq('total_credits')

pv = df5.loc[m, 'predicted_value'].map(to_money_decimal)
gt = df5.loc[m, 'ground_truth'   ].map(to_money_decimal)

# compare amounts (ignore sign); make sign matter by removing .abs()
new_acc = (pv.abs() == gt.abs()).astype('Int64')

# replace old accuracy with the new one for total_credits
df5.loc[m, 'accuracy'] = new_acc
