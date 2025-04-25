import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('table_analysis.csv')

# List of metrics to compare with response time
metrics = [
    'token_count', 'word_count', 'char_count', 'no_of_pages',
    'row_count', 'token_rate', 'tokens_generated', 'total_tokens'
]

# Set plot style
sns.set(style="whitegrid")

# Plot each metric against response time
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=metric, y='time', hue='time', palette='viridis', legend=False)
    plt.title(f'Response Time vs {metric}')
    plt.xlabel(metric)
    plt.ylabel('Response Time')
    plt.tight_layout()
    plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load the data
# Assuming the CSV has headers matching those in the table
df = pd.read_csv('table_analysis.csv')

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams.update({'font.size': 12})
plt.rcParams['figure.figsize'] = [14, 8]

# Define the metrics to analyze against response time
metrics = ['token_count', 'word_count', 'char_count', 'no_of_pages', 
           'row_count', 'token_rate', 'tokens_generated', 'total_tokens']

# Create a function to generate scatter plots with regression line
def plot_metric_vs_time(metric):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    scatter = sns.regplot(x=df[metric], y=df['time'], 
                         scatter_kws={'alpha':0.5, 's':80}, 
                         line_kws={'color':'red', 'linewidth':2},
                         ax=ax)
    
    # Calculate correlation coefficient
    corr, p_value = pearsonr(df[metric].dropna(), df['time'].dropna())
    
    # Add title and labels
    ax.set_title(f'{metric.replace("_", " ").title()} vs Response Time\nCorrelation: {corr:.3f} (p-value: {p_value:.4f})', 
                fontsize=16)
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel('Response Time (ms)', fontsize=14)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Improve aesthetics
    sns.despine()
    
    return fig

# Create a correlation heatmap
def plot_correlation_heatmap():
    # Select relevant columns
    corr_df = df[metrics + ['time']]
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    heatmap = sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                         linewidths=0.5, fmt=".2f", ax=ax)
    
    # Set title
    ax.set_title('Correlation Heatmap of Document Metrics', fontsize=16)
    
    return fig

# Generate scatter plots for each metric vs time
plots = {}
for metric in metrics:
    plots[metric] = plot_metric_vs_time(metric)
    plt.tight_layout()
    plt.savefig(f'{metric}_vs_time.png', dpi=300)
    plt.close()

# Create the correlation heatmap
heatmap = plot_correlation_heatmap()
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.close()

# Generate a summary report of key findings
def generate_summary_report():
    correlations = []
    
    for metric in metrics:
        corr, p_value = pearsonr(df[metric].dropna(), df['time'].dropna())
        correlations.append((metric, corr, p_value))
    
    # Sort by absolute correlation value (highest first)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("Summary Report: Correlations with Response Time")
    print("=" * 60)
    print(f"{'Metric':<20} {'Correlation':<15} {'p-value':<15} {'Significance':<15}")
    print("-" * 60)
    
    for metric, corr, p_value in correlations:
        significance = "High" if abs(corr) > 0.7 else "Medium" if abs(corr) > 0.4 else "Low"
        if p_value > 0.05:
            significance += " (not significant)"
            
        print(f"{metric.replace('_', ' ').title():<20} {corr:>+.3f}{'':>8} {p_value:.4f}{'':>7} {significance:<15}")
    
    print("=" * 60)

# Run the summary report
generate_summary_report()

# Create a combined visualization of top correlating factors
def plot_top_correlations():
    # Calculate correlations
    correlations = []
    for metric in metrics:
        corr, _ = pearsonr(df[metric].dropna(), df['time'].dropna())
        correlations.append((metric, abs(corr)))
    
    # Get top 4 metrics by absolute correlation
    top_metrics = sorted(correlations, key=lambda x: x[1], reverse=True)[:4]
    top_metric_names = [m[0] for m in top_metrics]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(top_metric_names):
        # Create scatter plot with regression line
        sns.regplot(x=df[metric], y=df['time'], 
                   scatter_kws={'alpha':0.6, 's':80}, 
                   line_kws={'color':'red', 'linewidth':2},
                   ax=axes[i])
        
        corr, p_value = pearsonr(df[metric].dropna(), df['time'].dropna())
        
        axes[i].set_title(f'{metric.replace("_", " ").title()} vs Response Time\nCorr: {corr:.3f}', 
                        fontsize=14)
        axes[i].set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        axes[i].set_ylabel('Response Time (ms)', fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('top_correlations.png', dpi=300)
    
    return fig

# Generate the top correlations plot
top_corr_plot = plot_top_correlations()
plt.close()

print("All visualizations have been generated successfully!")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import math

# Load the data
# Assuming the CSV has headers matching those in the table
df = pd.read_csv('table_analysis.csv')

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams.update({'font.size': 10})

# Define the metrics to analyze against response time
metrics = ['token_count', 'word_count', 'char_count', 'no_of_pages', 
           'row_count', 'token_rate', 'tokens_generated', 'total_tokens']

# Calculate number of rows and columns for the grid
num_metrics = len(metrics)
num_cols = 2  # Display two graphs side by side
num_rows = math.ceil(num_metrics / num_cols)  # Calculate required number of rows

# Create a single large figure with subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
axes = axes.flatten()  # Flatten to make indexing easier

# Generate a scatter plot for each metric
for i, metric in enumerate(metrics):
    if i < len(axes):  # Ensure we don't exceed the number of available axes
        ax = axes[i]
        
        # Create scatter plot with regression line
        sns.regplot(x=df[metric], y=df['time'], 
                   scatter_kws={'alpha':0.6, 's':60}, 
                   line_kws={'color':'red', 'linewidth':2},
                   ax=ax)
        
        # Calculate correlation coefficient
        corr, p_value = pearsonr(df[metric].dropna(), df['time'].dropna())
        
        # Add title and labels
        ax.set_title(f'{metric.replace("_", " ").title()} vs Response Time\nCorr: {corr:.3f} (p={p_value:.4f})', 
                    fontsize=14)
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Response Time (ms)', fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Improve aesthetics
        sns.despine(ax=ax)

# If there are any empty subplots, hide them
for i in range(num_metrics, len(axes)):
    axes[i].set_visible(False)

# Add overall title
plt.suptitle('Document Metrics vs Response Time Analysis', fontsize=20, y=1.02)

# Adjust layout
plt.tight_layout()
plt.savefig('all_metrics_vs_time.png', dpi=300)

# Create a correlation heatmap as a separate figure
plt.figure(figsize=(14, 12))
corr_df = df[metrics + ['time']]
corr_matrix = corr_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
           linewidths=0.5, fmt=".2f")
plt.title('Correlation Heatmap of Document Metrics', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)

# Generate a summary report of key findings
def generate_summary_report():
    correlations = []
    
    for metric in metrics:
        corr, p_value = pearsonr(df[metric].dropna(), df['time'].dropna())
        correlations.append((metric, corr, p_value))
    
    # Sort by absolute correlation value (highest first)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nSummary Report: Correlations with Response Time")
    print("=" * 60)
    print(f"{'Metric':<20} {'Correlation':<15} {'p-value':<15} {'Significance':<15}")
    print("-" * 60)
    
    for metric, corr, p_value in correlations:
        significance = "High" if abs(corr) > 0.7 else "Medium" if abs(corr) > 0.4 else "Low"
        if p_value > 0.05:
            significance += " (not significant)"
            
        print(f"{metric.replace('_', ' ').title():<20} {corr:>+.3f}{'':>8} {p_value:.4f}{'':>7} {significance:<15}")
    
    print("=" * 60)
    
    # Return the top correlating metrics
    return [x[0] for x in correlations[:4]]

# Run the summary report and get top metrics
top_metrics = generate_summary_report()

# Create an additional figure showing just the top correlating factors
fig_top, axes_top = plt.subplots(2, 2, figsize=(16, 12))
axes_top = axes_top.flatten()

for i, metric in enumerate(top_metrics):
    # Create scatter plot with regression line
    sns.regplot(x=df[metric], y=df['time'], 
               scatter_kws={'alpha':0.6, 's':80}, 
               line_kws={'color':'red', 'linewidth':2},
               ax=axes_top[i])
    
    corr, p_value = pearsonr(df[metric].dropna(), df['time'].dropna())
    
    axes_top[i].set_title(f'{metric.replace("_", " ").title()} vs Response Time\nCorr: {corr:.3f}', 
                    fontsize=14)
    axes_top[i].set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
    axes_top[i].set_ylabel('Response Time (ms)', fontsize=12)
    axes_top[i].grid(True, linestyle='--', alpha=0.7)

plt.suptitle('Top Correlating Factors with Response Time', fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig('top_correlations.png', dpi=300)

print("All visualizations have been generated successfully!")
print("\nThe following files have been created:")
print("1. all_metrics_vs_time.png - Combined visualization of all metrics")
print("2. correlation_heatmap.png - Heatmap showing correlations between all metrics")
print("3. top_correlations.png - Detail view of the top 4 correlating metrics")
