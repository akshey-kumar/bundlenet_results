import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Markov loss
learning_curves = []
for worm_num in range(5):
    algorithm = "BunDLeNet_temporal_shuffling"
    learning_curves_temporal_shuffling = np.load(f'data/generated/shuffling_experiments/learning_curves_{algorithm}_{worm_num}.npy', allow_pickle=True).item()
    algorithm = "BunDLeNet_no_shuffling"
    learning_curves_no_shuffling = np.load(f'data/generated/shuffling_experiments/learning_curves_{algorithm}_{worm_num}.npy', allow_pickle=True).item()
    plt.figure()
    plt.semilogy(learning_curves_temporal_shuffling["markov_test_loss"], label="temporal_shuffling")
    plt.semilogy(learning_curves_no_shuffling["markov_test_loss"], label="no_shuffling")
    plt.xlabel("Epochs")
    plt.ylabel("Markov loss $\mathcal{L}_M$")
    plt.legend()
    plt.tight_layout()
    learning_curves.append([learning_curves_temporal_shuffling["markov_test_loss"], learning_curves_no_shuffling["markov_test_loss"]])
learning_curves = np.array(learning_curves)
print(learning_curves, learning_curves.mean(axis=1).shape)

n_epochs = learning_curves.shape[2]
df = pd.DataFrame({
    'Epoch': np.tile(np.arange(1000), 5 * 2),
    "Markov Loss": learning_curves.flatten(),
    'Model': np.repeat(['BunDLeNet fitted on shuffled dynamics', 'BunDLeNet fitted on true dynamics'], 1000 * 5),
    'Worm': np.repeat(np.arange(5), 1000 * 2)
})

plt.figure(figsize=(10, 6))
epochs_plot = sns.lineplot(data=df, x='Epoch', y='Markov Loss', hue='Model', ci='sd')
epochs_plot.set(yscale='log')
plt.show()




# Behaviour loss
learning_curves = []
for worm_num in range(5):
    algorithm = "BunDLeNet_behaviour_shuffling"
    learning_curves_behaviour_shuffling = np.load(f'data/generated/shuffling_experiments/learning_curves_{algorithm}_{worm_num}.npy', allow_pickle=True).item()
    algorithm = "BunDLeNet_no_shuffling"
    learning_curves_no_shuffling = np.load(f'data/generated/shuffling_experiments/learning_curves_{algorithm}_{worm_num}.npy', allow_pickle=True).item()
    plt.figure()
    plt.semilogy(learning_curves_behaviour_shuffling["behaviour_train_loss"], label="behaviour_shuffling")
    plt.semilogy(learning_curves_no_shuffling["behaviour_train_loss"], label="no_shuffling")
    plt.xlabel("Epochs")
    plt.ylabel("behaviour loss $\mathcal{L}_B$")
    plt.legend()
    plt.tight_layout()
    learning_curves.append([learning_curves_behaviour_shuffling["behaviour_train_loss"], learning_curves_no_shuffling["behaviour_train_loss"]])
learning_curves = np.array(learning_curves)
print(learning_curves, learning_curves.mean(axis=1).shape)

n_epochs = learning_curves.shape[2]
df = pd.DataFrame({
    'Epoch': np.tile(np.arange(1000), 5 * 2),
    "behaviour Loss": learning_curves.flatten(),
    'Model': np.repeat(['BunDLeNet fitted on shuffled behaviour', 'BunDLeNet fitted on true behaviour'], 1000 * 5),
    'Worm': np.repeat(np.arange(5), 1000 * 2)
})

plt.figure(figsize=(10, 6))
epochs_plot = sns.lineplot(data=df, x='Epoch', y='behaviour Loss', hue='Model', ci='sd')
epochs_plot.set(yscale='log')
plt.show()
