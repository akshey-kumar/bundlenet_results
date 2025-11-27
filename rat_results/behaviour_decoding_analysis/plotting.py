import numpy as np
import matplotlib.pyplot as plt

rat_name = "achilles"
algorithms = ["bundlenet", "cca_tde", "cebra_hybrid"]

# Soft, scientifically neutral colors
colors = {
    "bundlenet": "#0072B2",     # strong blue (your method)
    "cca_tde": "#E69F00",       # soft orange
    "cebra_hybrid": "#009E73",  # soft green
}

fig, axes = plt.subplots(1, 3, figsize=(9, 2), sharey=True)

for ax, algo in zip(axes, algorithms):

    file_pattern = (
        f"data/generated/predicted_and_true_behaviours/{{}}__{algo}_rat_{rat_name}"
    )
    b_test_pred = np.loadtxt(file_pattern.format("b_test_1_pred"))
    b_test_true = np.loadtxt(file_pattern.format("b_test_1"))

    # Emphasis: BunDLe-Net looks best
    if algo == "bundlenet":
        ax.plot(
            b_test_pred[:, 0],
            color=colors[algo],
            linewidth=1.6,
            alpha=1.0,
            label="predicted",
            zorder=5,
        )
    else:
        ax.plot(
            b_test_pred[:, 0],
            color=colors[algo],
            linewidth=1.0,
            alpha=0.8,
            label="predicted",
            zorder=3,
        )

    ax.plot(
        b_test_true[:, 0],
        color="black",
        linewidth=1.0,
        alpha=0.9,
        label="true",
    )

    ax.set_title(algo.replace("_", " "), pad=4)
    ax.set_xticks([])
    ax.margins(x=0)

    # Legend with same font size as rest of figure
    ax.legend(
        frameon=False,
        fontsize="medium",   # matches default text size
        loc="upper right",
        handlelength=1.3
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("position")
for ax in axes:
    ax.set_xlabel("time")

fig.tight_layout()
plt.show()

fig.savefig('rat_results/behaviour_decoding_analysis/behaviour_comparison.png', dpi=1200, bbox_inches='tight', transparent=True, pad_inches=0.01)

exit()
import numpy as np
import matplotlib.pyplot as plt

rat_name = "achilles"
algorithms = ["bundlenet", "cca_tde", "cebra_hybrid"]

# Soft, scientifically neutral colors
colors = {
    "bundlenet": "#0072B2",     # strong blue (your method)
    "cca_tde": "#E69F00",       # soft orange
    "cebra_hybrid": "#009E73",  # soft green
}

fig, axes = plt.subplots(1, 3, figsize=(9, 2), sharey=True)

for ax, algo in zip(axes, algorithms):

    file_pattern = f"data/generated/predicted_and_true_behaviours/{{}}__{algo}_rat_{rat_name}"
    b_test_pred = np.loadtxt(file_pattern.format("b_test_1_pred"))
    b_test_true = np.loadtxt(file_pattern.format("b_test_1"))

    # Emphasis: BunDLe-Net looks best
    if algo == "bundlenet":
        ax.plot(
            b_test_pred[:, 0],
            color=colors[algo],
            linewidth=1.6,
            alpha=1.0,
            label="predicted",
            zorder=5,
        )
    else:
        ax.plot(
            b_test_pred[:, 0],
            color=colors[algo],
            linewidth=1.0,
            alpha=0.8,
            label="predicted",
            zorder=3,
        )

    # Ground truth always the same for comparison
    ax.plot(
        b_test_true[:, 0],
        color="black",
        linewidth=1.0,
        alpha=0.9,
        label="true",
    )

    # Aesthetics
    ax.set_title(algo.replace("_", " "), pad=4)
    ax.set_xticks([])
    ax.margins(x=0)

    # Clean panel look
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

# Shared axis label
axes[0].set_ylabel("position")
for ax in axes:
    ax.set_xlabel("time")

# One unified legend (NM style: below the row)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False,
    fontsize=8,
    handlelength=1.5,
    bbox_to_anchor=(0.5, -0.15),
)

fig.tight_layout()
plt.show()
fig.savefig('rat_results/behaviour_decoding_analysis/behaviour_comparison.png', dpi=1200, bbox_inches='tight', transparent=True, pad_inches=0.01)


exit()
import sys
import numpy as np
import matplotlib.pyplot as plt

rat_name = 'achilles' #'achilles', 'gatsby', 'cicero', 'buddy'


for algorithm in ['bundlenet', 'cca_tde', 'cebra_hybrid']:
    plt.figure(figsize=(6, 2))
    file_pattern = f'data/generated/predicted_and_true_behaviours/{{}}__{algorithm}_rat_{rat_name}'
    b_train_1_pred = np.loadtxt(file_pattern.format('b_train_1_pred'))
    b_test_1_pred = np.loadtxt(file_pattern.format('b_test_1_pred'))
    b_train_1 = np.loadtxt(file_pattern.format('b_train_1'))
    b_test_1 = np.loadtxt(file_pattern.format('b_test_1'))

    plt.plot(b_test_1_pred[:,0], label=f'{algorithm} predicted behaviour')
    plt.ylabel('position')
    plt.xlabel('time')


    plt.plot(b_test_1[:,0], label='True behaviour')
    plt.xticks([])
    plt.legend()
    plt.show()

'''
plt.figure(figsize=(10,3))
plt.plot(b_train_1_pred[:,0], label=f'{algorithm} predicted behaviour')
plt.plot(b_train_1[:,0], label='True behaviour')
plt.ylabel('position')
plt.xlabel('time')
plt.legend()
plt.show()
'''