import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────────────────────────
# 1) Replace these dummy lists with your actual label Series. For example:
#       y_true = results_df['SOR_Decision']
#       y_pred = results_df['model_decision']
#
#    Make sure they contain the exact same label set (e.g. "yes" / "no").
# ─────────────────────────────────────────────────────────────────────────────

# Example dummy data (replace these with your real data!)
y_true = ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'no']
y_pred = ['yes', 'no', 'no',  'no', 'yes', 'yes','yes', 'no',  'no']

# ─────────────────────────────────────────────────────────────────────────────
# 2) Define the label‐order and (optionally) human‐readable names:
# ─────────────────────────────────────────────────────────────────────────────
labels = ['yes', 'no']
target_names = ['Belongs (yes)', 'Doesn’t Belong (no)']

# ─────────────────────────────────────────────────────────────────────────────
# 3) Build a confusion matrix (NumPy array) and wrap it in a DataFrame
# ─────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(
    cm,
    index=[f"True: {lab}" for lab in labels],
    columns=[f"Pred: {lab}" for lab in labels]
)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Build a classification report (as a pandas DataFrame):
# ─────────────────────────────────────────────────────────────────────────────
report_dict = classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=target_names,
    output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()

# ─────────────────────────────────────────────────────────────────────────────
# 5) Plot & save the confusion matrix DataFrame as an image:
# ─────────────────────────────────────────────────────────────────────────────
fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
ax_cm.axis('off')  # hide the axes

# Create a table from the DataFrame:
tbl_cm = ax_cm.table(
    cellText=cm_df.values,
    rowLabels=cm_df.index,
    colLabels=cm_df.columns,
    cellLoc='center',
    loc='center'
)
tbl_cm.auto_set_font_size(False)
tbl_cm.set_fontsize(10)
tbl_cm.scale(1.2, 1.2)  # scale (width, height) to make room

fig_cm.tight_layout()
fig_cm.savefig("confusion_matrix.png", dpi=200, bbox_inches='tight')
plt.close(fig_cm)  # close the figure to free memory

# ─────────────────────────────────────────────────────────────────────────────
# 6) Plot & save the classification report DataFrame as an image:
# ─────────────────────────────────────────────────────────────────────────────
fig_cr, ax_cr = plt.subplots(figsize=(6, 4))
ax_cr.axis('off')  # hide the axes

tbl_cr = ax_cr.table(
    cellText=report_df.round(2).values,
    rowLabels=report_df.index,
    colLabels=report_df.columns,
    cellLoc='center',
    loc='center'
)
tbl_cr.auto_set_font_size(False)
tbl_cr.set_fontsize(9)
tbl_cr.scale(1.2, 1.2)

fig_cr.tight_layout()
fig_cr.savefig("classification_report.png", dpi=200, bbox_inches='tight')
plt.close(fig_cr)

print("Saved confusion matrix ▶ confusion_matrix.png")
print("Saved classification report ▶ classification_report.png")
