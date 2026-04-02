import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split

algorithm = 'BunDLeNet'
rat_name = 'achilles'
# Load data
data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
x, b = data['x'], data['b']
x = x - np.min(x)

os.makedirs('data/generated/bunlenet_parameter_experiments', exist_ok=True)

results = []
for latent_dim in range(1,11):
    for i in range(5):
        print(f"latent_dim: {latent_dim}, iteration:{i}")
        x_, b_ = prep_data(x, b, win=20)
        print(x_.shape, b_.shape)
        model = BunDLeNet(latent_dim=latent_dim, num_behaviour=b_.shape[1], input_shape=x_.shape)
        history, _ = train_model(
            x_,
            b_,
            model,
            b_type='continuous',
            gamma=0.9,
            learning_rate=0.001,
            n_epochs=500,
        )
        results.append({
            "latent_dim": latent_dim,
            "markov_loss": history[-1,0],
            "behaviour_loss": history[-1, 1],
            "total_loss": history[-1,-1],

        })

        np.save(f'data/generated/bunlenet_parameter_experiments/losses_vs_latent_dim_{algorithm}_rat_{rat_name}.npy', results)


# Plotting
results = np.load(f'data/generated/bunlenet_parameter_experiments/losses_vs_latent_dim_{algorithm}_rat_{rat_name}.npy', allow_pickle=True)
df = pd.DataFrame.from_dict(list(results))
print(df.head())

df_melted = df.melt(id_vars='latent_dim', value_vars=['markov_loss', 'behaviour_loss'],
                    var_name='Loss Type', value_name='Loss')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_melted, x='latent_dim', y='Loss', hue='Loss Type', style='Loss Type', s=100)
plt.xlabel('Latent Dimension')
plt.ylabel('Train Loss')
plt.title('Markov and Behaviour Loss vs Latent Dimension')
plt.show()