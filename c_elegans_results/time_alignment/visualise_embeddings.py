import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

b_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
b_names = {idx:name for idx, name in enumerate(b_names)}
algorithm = 'BunDLeNet'

for worm_num in range(5):
    y0_ = np.loadtxt(f'data/generated/embeddings/c_elegans/Y0_tr__{algorithm}_worm_{worm_num}')
    b_ = np.loadtxt(f'data/generated/embeddings/c_elegans/B_train_1__{algorithm}_worm_{worm_num}').astype(int)
    print(algorithm)

    # Plotting latent space dynamics
    vis = LatentSpaceVisualiser(
        y0_,
        b_,
        b_names,
        show_points=False,
        legend=False,
    )
    #vis.plot_latent_timeseries()
    vis.plot_phase_space(
        show_fig=True,
        arrow_length_ratio=0.1)
    #vis.rotating_plot(
    #    filename=f'c_elegans_results/embedding_algorithms/figures/rotation_{algorithm}_worm_{worm_num}.gif',
    #    arrow_length_ratio=0.1
    #)
#plt.show()
