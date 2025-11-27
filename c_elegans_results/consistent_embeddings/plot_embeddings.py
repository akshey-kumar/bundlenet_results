import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

algorithm = 'bundlenet'
b_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
b_names = {idx:name for idx, name in enumerate(b_names)}
for worm_num in range(5):
    y0_ = np.loadtxt(f'data/generated/embeddings/c_elegans/comparable_embeddings/Y0__{algorithm}_worm_{worm_num}')
    b_ = np.loadtxt(f'data/generated/embeddings/c_elegans/comparable_embeddings/B__{algorithm}_worm_{worm_num}').astype(int)
    print(algorithm)
    deep_palette = sns.color_palette('deep', 8)
    colors = [deep_palette[i] for i in [4, 2, 7, 5, 1, 3, 0, 6]]

    vis = LatentSpaceVisualiser(
        y0_,
        b_,
        b_names,
        colors=colors,
        show_points=False,
        legend=True,
    )
    #vis.plot_latent_timeseries()
    #vis.plot_phase_space(
    #    show_fig=False,
    #    arrow_length_ratio=0.4)
    os.makedirs('c_elegans_results/consistent_embeddings/figures/', exist_ok=True)
    vis.rotating_plot(
        filename=f'c_elegans_results/consistent_embeddings/figures/rotation_{algorithm}_worm_{worm_num}.gif',
        arrow_length_ratio=0.1,
        show_fig=False,
    )
plt.show()
