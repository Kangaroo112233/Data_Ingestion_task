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
