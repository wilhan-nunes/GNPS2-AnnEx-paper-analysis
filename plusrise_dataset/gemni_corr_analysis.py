import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
filename = "annex_summary_b22ff1bc4162 (8).csv"
df = pd.read_csv(filename)

# Set up the 2x2 figure grid
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Define features, titles, and colors for the 4 panels
features = [
    {
        'x': 'mz_precision', 
        'y': 'confidence_score', 
        'ax': axes[0, 0], 
        'scatter_color': 'royalblue', 
        'line_color': 'navy'
    },
    {
        'x': 'median_abs_ppm_error', 
        'y': 'confidence_score', 
        'ax': axes[0, 1], 
        'scatter_color': 'coral', 
        'line_color': 'maroon'
    },
    {
        'x': 'supporting_matches', 
        'y': 'confidence_score', 
        'ax': axes[1, 0], 
        'scatter_color': 'orchid', 
        'line_color': 'purple'
    },
    {
        'x': 'tanimoto_score', 
        'y': 'confidence_score', 
        'ax': axes[1, 1], 
        'scatter_color': 'mediumseagreen', 
        'line_color': 'darkgreen'
    }
]

# Loop through and generate each subplot
for f in features:
    ax = f['ax']
    x_col = f['x']
    y_col = f['y']
    
    # Calculate Pearson correlation coefficient
    r_val = df[x_col].corr(df[y_col])
    
    # Plot scatter points with low alpha for density visualization
    sns.scatterplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        ax=ax,
        color=f['scatter_color'], 
        alpha=0.3,
        edgecolor=None
    )
    
    # Overlay OLS linear regression trendline with 95% CI bands
    sns.regplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        ax=ax,
        scatter=False, 
        color=f['line_color'], 
        line_kws={'linewidth': 2}
    )
    
    # Customize titles, labels, and grid
    ax.set_title(f'All Scans\n{x_col} vs Confidence (r = {r_val:.2f})', fontsize=12)
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

# Adjust spacing and save the figure
plt.tight_layout()
plt.savefig('correlation_analysis_4panel_figure.png', dpi=300, bbox_inches='tight')
plt.show()