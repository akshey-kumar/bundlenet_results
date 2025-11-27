import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

b_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
b_names = {idx:name for idx, name in enumerate(b_names)}
algorithms = ['BunDLeNet_behaviour_shuffling', 'BunDLeNet_temporal_shuffling']
worm_num = 0
for algorithm in algorithms:
    y0_ = np.loadtxt(f'data/generated/shuffling_experiments/Y0__{algorithm}_worm_{worm_num}')
    b_ = np.loadtxt(f'data/generated/shuffling_experiments/B__{algorithm}_worm_{worm_num}').astype(int)
    print(worm_num, algorithm, y0_,b_)
    #deep_palette = sns.color_palette('deep', 8)
    #colors = [deep_palette[i] for i in [4, 2, 7, 5, 1, 3, 0, 6]]

    # Plotting latent space dynamics
    vis = LatentSpaceVisualiser(
        y0_,
        b_,
        b_names,
        show_points=True,
        legend=False,
    )
    #vis.plot_latent_timeseries()
    fig, ax = vis.plot_phase_space(
        arrow_length_ratio=0.1,
        show_fig=False
    )
    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=-1, top=1)
    plt.show()
    #vis.rotating_plot(
    #    filename=f'c_elegans_results/embedding_algorithms/figures/rotation_{algorithm}_worm_{worm_num}.gif',
    #    arrow_length_ratio=0.1
    #)
#plt.show()
