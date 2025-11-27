import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
import sys

algorithm = sys.argv[1]
worm_num = sys.argv[2]
state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']

file_pattern = f'data/generated/quantitative_evaluation/embeddings/c_elegans/{{}}__{algorithm}_worm_{worm_num}'
y0_tr = np.loadtxt(file_pattern.format('y0_tr'))
y1_tr = np.loadtxt(file_pattern.format('y1_tr'))
y0_tst = np.loadtxt(file_pattern.format('y0_tst'))
y1_tst = np.loadtxt(file_pattern.format('y1_tst'))
b_tr = np.loadtxt(file_pattern.format('b_tr')).astype(int)
b_tst = np.loadtxt(file_pattern.format('b_tst')).astype(int)

# Discrete variable plotting
deep_palette = sns.color_palette('deep', 8)
colors = [deep_palette[i] for i in [4, 2, 7, 5, 1, 3, 0, 6]]



fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')

vis = LatentSpaceVisualiser(
    y=y0_tr,
    b=b_tr,
    b_names=state_names,
    colors=colors,
)
fig, ax = vis._plot_ps(fig, ax, arrow_length_ratio=0.001, alpha=0.5)

# vis = LatentSpaceVisualiser(
#     y=y0_tst,
#     b=b_tst,
#     b_names=state_names,
#     colors=colors,
#     show_points=False
# )
# fig, ax = vis._plot_ps(fig, ax, arrow_length_ratio=0.0001)

ax.scatter(y0_tst[:, 0], y0_tst[:, 1], y0_tst[:, 2], c=[colors[i] for i in b_tst], s=10)
plt.show()

