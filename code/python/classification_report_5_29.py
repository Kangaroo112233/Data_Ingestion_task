from sklearn.metrics import classification_report, confusion_matrix

# if your columns are already strings “yes”/“no” you can pass them directly:
y_true = results_df['SOR_Decision']
y_pred = results_df['model_decision']

print("Confusion matrix:\n",
      confusion_matrix(y_true, y_pred, labels=['yes','no']))

print("\nClassification report:\n",
      classification_report(y_true, y_pred,
                            labels=['yes','no'],
                            target_names=['belongs (yes)','doesn’t belong (no)']))
