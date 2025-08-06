def get_metrics_table(results_df):
    field_names = results_df['field'].unique()

    correct = {tag: results_df[(results_df['field'] == tag) & (results_df['is_correct'])].shape[0] for tag in field_names}
    support = {tag: results_df[results_df['field'] == tag].shape[0] for tag in field_names}
    accuracy = {tag: round(correct[tag] / support[tag] * 100, 2) if support[tag] > 0 else 0.0 for tag in field_names}

    # Add overall values
    correct["Overall"] = sum(correct.values())
    support["Overall"] = sum(support.values())
    accuracy["Overall"] = round(correct["Overall"] / support["Overall"] * 100, 2)

    metrics_df = pd.DataFrame.from_dict({
        "correct": correct,
        "support": support,
        "accuracy": accuracy
    })

    return metrics_df

metrics_table = get_metrics_table(df)
print(metrics_table)



# Step 1: Define the 6 fields you want to include
selected_fields = [
    'client_name',
    'account_number',
    'total_charges',
    'total_credits',
    'due_date',
    'statement_end_date'
]

# Step 2: Filter df4 to only include selected fields
filtered_df = df4[df4['field'].isin(selected_fields)].copy()

# Step 3: Compute field-wise accuracy
metrics_table = (
    filtered_df.groupby('field')['accuracy']
    .agg(['sum', 'count'])
    .rename(columns={'sum': 'correct', 'count': 'support'})
)
metrics_table['accuracy'] = (metrics_table['correct'] / metrics_table['support']) * 100
metrics_table = metrics_table.reset_index()

# Step 4: Compute overall accuracy for just the selected fields
overall_correct = filtered_df['accuracy'].astype(int).sum()
overall_total = len(filtered_df)
overall_accuracy = (overall_correct / overall_total) * 100

# Step 5: Append overall row
overall_row = pd.DataFrame([{
    'field': 'Overall',
    'correct': overall_correct,
    'support': overall_total,
    'accuracy': overall_accuracy
}])

metrics_table = pd.concat([metrics_table, overall_row], ignore_index=True)

# Step 6: Print the result
print(metrics_table)

