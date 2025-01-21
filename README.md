# Data_Ingestion_task


import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

# Example data
y_true = np.array([0, 1, 1, 0, 1, 1, 0])  # True labels
y_probs = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.6, 0.2])  # Predicted probabilities

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

# Interactive Plot Function
def interactive_roc(threshold):
    # Find the index of the closest threshold
    idx = np.argmin(np.abs(thresholds - threshold))
    selected_fpr = fpr[idx]
    selected_tpr = tpr[idx]
    
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    
    # Highlight the selected threshold point
    plt.scatter([selected_fpr], [selected_tpr], color="red", label=f"Threshold = {threshold:.2f}\nFPR = {selected_fpr:.2f}, TPR = {selected_tpr:.2f}", zorder=5)
    
    # Add labels and legend
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Interactive ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# Create a slider for threshold selection
threshold_slider = FloatSlider(
    value=0.5,
    min=min(thresholds),
    max=max(thresholds),
    step=0.01,
    description='Threshold:',
    continuous_update=True
)

# Use `interact` to link the slider to the plot
interact(interactive_roc, threshold=threshold_slider)
