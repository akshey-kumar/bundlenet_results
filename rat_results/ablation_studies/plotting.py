import os

import numpy as np
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
import matplotlib.pyplot as plt

algorithm = 'BunDLeNet_mixed_model'
rat_names = [ 'achilles' , 'gatsby','cicero', 'buddy']
for rat_name in rat_names:
    for gamma in [0, 0.9, 1]: #[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 1]:
        y0_ = np.loadtxt(f'data/generated/embeddings/rat/ablation_experiments/y0__{algorithm}_rat_{rat_name}_gamma_{gamma}')
        b_ = np.loadtxt(f'data/generated/embeddings/rat/ablation_experiments/b__{algorithm}_rat_{rat_name}_gamma_{gamma}')
        print(rat_name, gamma)
        print(b_)
        # Plotting latent space dynamics

        # Continuous variable plotting
        fig = plt.figure(figsize=(4, 4))
        ax = plt.axes(projection='3d')
        ax.axis('off')
        pts = ax.scatter(y0_[:, 0], y0_[:, 1], y0_[:, 2], c=b_[:, 0], s=0.5)
        plt.colorbar(pts)
        plt.colorbar(pts)
        os.makedirs(f'rat_results/ablation_studies/figures/', exist_ok=True)
        gamma_str = str(gamma).replace('.', 'p')  # replace dot so filename has no extra '.' parts
        print(gamma_str)
        plt.savefig(
            f'rat_results/ablation_studies/figures/rat_{rat_name}_embedding_{algorithm}_ablation_gamma_{gamma_str}_position_labels',
            transparent=True, bbox_inches="tight", dpi=300)

        # Discrete variable plotting
        vis = LatentSpaceVisualiser(
            y=y0_,
            b=b_[:, 1].astype(int),
            b_names=['0', '1']
        )
        vis.plot_phase_space(
            show_fig=False,
            arrow_length_ratio=0.01,
        )
        plt.savefig(
            f'rat_results/ablation_studies/figures/rat_{rat_name}_embedding_{algorithm}_ablation_gamma_{gamma_str}_direction_labels',
            transparent=True, bbox_inches="tight", dpi=300)
