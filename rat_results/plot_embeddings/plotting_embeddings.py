import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser


algorithm = 'bundlenet'
for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    # Load data
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')

    y0_ = np.loadtxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}')
    b_ = np.loadtxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}')

    # Continuous variable plotting
    fig = plt.figure(figsize=(4, 4))
    ax = plt.axes(projection='3d')
    ax.axis('off')
    pts = ax.scatter(y0_[:, 0], y0_[:, 1], y0_[:, 2], c=b_[:, 0], s=0.5)
    plt.colorbar(pts)

    # Discrete variable plotting
    vis = LatentSpaceVisualiser(
        y=y0_,
        b=b_[:, 1].astype(int),
        b_names=['0', '1']
    )
    vis.plot_phase_space(
       show_fig=False,
       arrow_length_ratio=0.00001,
    )

    plt.show()
