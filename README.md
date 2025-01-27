import numpy as np
import matplotlib.pyplot as plt

def compute_ece(y_true, y_prob, num_bins=10):
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    
    ece = 0
    for i in range(num_bins):
        mask = bin_indices == i
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_size = np.sum(mask)
            ece += (bin_size / len(y_true)) * abs(bin_acc - bin_conf)
    return ece

def prepare_reliability_data(y_true, y_prob, num_bins=10):
    """Prepare data for reliability diagram."""
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    accuracies = np.zeros(num_bins)
    confidences = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = bin_indices == i
        if np.any(mask):
            accuracies[i] = np.mean(y_true[mask])
            confidences[i] = np.mean(y_prob[mask])
            counts[i] = np.sum(mask)
            
    return accuracies, confidences, counts, bin_edges, bin_centers

def plot_reliability_diagram_41(ax1, ax2, y_true, y_prob, num_bins=10, title_prefix=""):
    """Plot reliability diagram with bars showing calibration."""
    # Compute ECE
    ece_value = compute_ece(y_true, y_prob, num_bins=num_bins)
    
    # Get bin edges and centers
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate accuracies and counts for each bin
    accuracies = np.zeros(num_bins)
    confidences = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = bin_indices == i
        if np.any(mask):
            accuracies[i] = np.mean(y_true[mask])
            confidences[i] = np.mean(y_prob[mask])
            counts[i] = np.sum(mask)
    
    # Calculate bar width
    bar_width = bin_edges[1] - bin_edges[0]
    
    # Plot reliability diagram (top plot)
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    
    # Bars for outputs (blue)
    ax1.bar(bin_centers, accuracies, width=bar_width, alpha=0.8,
            color='royalblue', label='Outputs', edgecolor='black')
    
    # Calculate and plot calibration gaps
    gap = accuracies - bin_centers
    gap_positive = gap > 0
    gap_negative = gap < 0
    
    # Plot positive gaps
    if np.any(gap_positive):
        ax1.bar(bin_centers[gap_positive], gap[gap_positive],
                bottom=bin_centers[gap_positive], width=bar_width,
                alpha=0.5, color='salmon', hatch='//', edgecolor='red',
                label='Calibration Gap')
    
    # Plot negative gaps
    if np.any(gap_negative):
        ax1.bar(bin_centers[gap_negative], -gap[gap_negative],
                bottom=accuracies[gap_negative], width=bar_width,
                alpha=0.5, color='salmon', hatch='//', edgecolor='red')
    
    # Set labels and limits for reliability diagram
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='lower right')
    
    # Plot count histogram (bottom plot)
    ax2.bar(bin_centers, counts, width=bar_width, alpha=0.7,
            color='steelblue', edgecolor='black')
    ax2.set_xlabel('Predicted Probabilities')
    ax2.set_ylabel('Count')
    ax2.grid(axis='y')
    
    # Title with ECE
    plt.suptitle(f'{title}\nECE={ece_value:.3f}, bins={num_bins}')
    plt.tight_layout()
    
    return fig, (ax1, ax2)

# Example usage:
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulated probabilities and true labels
    y_prob = np.random.beta(5, 2, n_samples)
    y_true = np.random.binomial(1, y_prob)
    
    # Plot reliability diagram
    fig, axes = plot_reliability_diagram(
        y_true, y_prob,
        title="Sample Reliability Diagram",
        num_bins=10
    )
    plt.show()
