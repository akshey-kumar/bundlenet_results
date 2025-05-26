import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser


worm_num = 0
b_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
b_names = {idx:name for idx, name in enumerate(b_names)}
for algorithm in [ 'bundlenet', 'lda', 'pca','tsne_optimised', 'autoencoder_optimised', 'autoregressor_autoencoder_optimised', 'cebra_hybrid_optimised']:
    # Plotting
    y0_ = np.loadtxt(f'data/generated/embeddings/y0__{algorithm}_worm_{worm_num}')
    b_ = np.loadtxt(f'data/generated/embeddings/b__{algorithm}_worm_{worm_num}').astype(int)
    print(algorithm)
    deep_palette = sns.color_palette('deep', 8)
    colors = [deep_palette[i] for i in [4, 2, 7, 5, 1, 3, 0, 6]]

    # Plotting latent space dynamics
    vis = LatentSpaceVisualiser(
        y0_,
        b_,
        b_names,
        colors=colors,
        show_points=False,
        legend=False,
    )
    #vis.plot_latent_timeseries()
    vis.plot_phase_space(show=False)
    # vis.rotating_plot(filename=f'figures/rotation_{algorithm}_worm_{worm_num}.gif')
plt.show()
