import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from behaviour_alignment import behaviour_alignment
import seaborn as sns

algorithms = ['bundlenet', 'bundlenet_linear', 'cebra_hybrid', 'pca_tde', 'cca_tde', 'rrr_tde']
for algorithm in algorithms:
    rat_names = ['achilles', 'gatsby', 'cicero', 'buddy']
    reg_scores = np.zeros((len(rat_names), len(rat_names)))

    for i, rat_name_i in enumerate(rat_names):
        for j, rat_name_j in enumerate(rat_names):

            y_t_i = np.loadtxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name_i}')
            b_t_i = np.loadtxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name_i}')
            y_t_j = np.loadtxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name_j}')
            b_t_j = np.loadtxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name_j}')

            y_b_i = behaviour_alignment(y_t_i, b_t_i, n_bins = 1600)
            y_b_j = behaviour_alignment(y_t_j, b_t_j, n_bins = 1600)

            df = pd.merge(y_b_i, y_b_j, on='b', how = 'inner', suffixes=('_i','_j'))

            y_i = np.stack(df['y_i'].to_numpy())
            y_j = np.stack(df['y_j'].to_numpy())

            # Regressing y_j (dependent variable) on y_i (independent variable)
            reg = LinearRegression().fit(y_i, y_j)
            reg_scores[i,j] = reg.score(y_i, y_j)

    # Create a heatmap with annotations

    font_size = 12  # choose the desired uniform font size

    plt.figure(figsize=(4, 3))
    # apply global font size for matplotlib elements
    plt.rcParams.update({'font.size': font_size})

    # Plot the heatmap
    ax = sns.heatmap(reg_scores, annot=True, fmt=".2f", cmap='YlGnBu',
                cbar_kws={'label': 'Regression Score'},
                xticklabels=['rat 1', 'rat 2', 'rat 3', 'rat 4'],
                yticklabels=['rat 1', 'rat 2', 'rat 3', 'rat 4'],
                vmax=0, vmin=1,
                linewidths=0.5, linecolor='gray',
                annot_kws={'fontsize': font_size}
                )
    # ensure tick labels use the same font size
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    # Set the title and show the plot
    plt.title(algorithm, fontsize=font_size)
    plt.tight_layout()
    os.makedirs('rat_results/consistency_of_behaviour_aligned_embeddings/figures', exist_ok=True)
    plt.savefig(f'rat_results/consistency_of_behaviour_aligned_embeddings/figures/consistency_plot_{algorithm}', dpi=300,bbox_inches='tight', pad_inches=0.02)
plt.show()
