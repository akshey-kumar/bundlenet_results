import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

b_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
b_names = {idx:name for idx, name in enumerate(b_names)}
algorithms = ['bundlenet_ablation_study_purely_behavioural_worm_0_gamma_0.0', 'bundlenet_ablation_study_purely_dynamical_worm_0_gamma_1']
worm_num = 0
for algorithm in algorithms:
    y0_ = np.loadtxt(f'data/generated/ablation_studies/y0__{algorithm}')
    b_ = np.loadtxt(f'data/generated/ablation_studies/b__{algorithm}').astype(int)
    print(algorithm, y0_,b_)
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
    vis.plot_phase_space(
        arrow_length_ratio=0.1)
    #vis.rotating_plot(
    #    filename=f'c_elegans_results/embedding_algorithms/figures/rotation_{algorithm}_worm_{worm_num}.gif',
    #    arrow_length_ratio=0.1
    #)
#plt.show()
'''
###################################
# Plotting varying gamma
b_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
b_names = {idx:name for idx, name in enumerate(b_names)}
algorithm = 'bundlenet_ablation_study'
worm_num = 0
for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.99, 0.999]:
    y0_ = np.loadtxt(f'data/generated/ablation_studies/y0__{algorithm}_worm_{worm_num}_gamma_{gamma}')
    b_ = np.loadtxt(f'data/generated/ablation_studies/b__{algorithm}_worm_{worm_num}_gamma_{gamma}').astype(int)
    print(gamma, y0_,b_)
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
'''