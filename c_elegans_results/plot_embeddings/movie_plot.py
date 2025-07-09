import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

b_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
b_names = {0: 'forward', 1: 'reverse', 2: 'sustained reversal', 3: 'turn'}
#b_names = {idx:name for idx, name in enumerate(b_names)}
worm_num = 0
algorithm = 'bundlenet'

# Plotting
y0_ = np.loadtxt(f'c_elegans_results/plot_embeddings/model_multistep_2_worm6_Y0')
b_ = np.loadtxt(f'c_elegans_results/plot_embeddings/model_multistep_2_worm6_B').astype(int)
print(algorithm)
deep_palette = sns.color_palette('deep', 8)
colors = [deep_palette[i] for i in [4, 2, 7, 5, 1, 3, 0, 6]]

# Plotting latent space dynamics
vis = LatentSpaceVisualiser(
    y0_[:],
    b_[:],
    b_names,
    #colors=colors,
    #show_points=False,
    #legend=False,
)

#vis.plot_phase_space(
#    show_fig=False,
#    arrow_length_ratio=0.4)
ret = vis.make_movie(
    fps=30,
    filename='c_elegans_results/plot_embeddings/worm_6_movie.gif',
    show_fig=False,
    # initial_alpha=0.1, # for making background manifold visible
    # fade_time=10, # how many last frames will be displayed
)
