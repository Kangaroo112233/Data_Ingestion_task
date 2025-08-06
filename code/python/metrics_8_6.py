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


