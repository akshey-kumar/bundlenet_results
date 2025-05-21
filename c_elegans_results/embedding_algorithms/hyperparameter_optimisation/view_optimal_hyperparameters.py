import pandas as pd
from ray import tune
#from c_elegans_results.algorithms.hyperparameter_optimisation.autoencoder import train_autoencoder
from autoencoder import train_autoencoder

# Define the log directory where results are saved
logdir = '/home/akshey/ray_results/train_autoencoder_2025-02-13_14-56-13'

restored_tuner = tune.Tuner.restore(logdir, trainable=train_autoencoder)
result_grid = restored_tuner.get_results()
df = result_grid.get_dataframe()

# autoencoder architecture
# Autoencoder architecture function
def architecture_from_idx(idx):
    architectures = [
        [50, 10],               # Shallow Architecture
        [100, 150, 50, 10],     # Increasing then Decreasing Architecture
        [50, 30, 25, 10],       # Same encoder as in BunDLeNet
        [100, 80, 60, 40, 20, 10, 10],  # Deep Architecture
        [75, 25, 10]
    ]
    return architectures[int(idx)]

# Apply the architecture function element-wise
df['config/layers'] = df['config/layers_idx'].apply(architecture_from_idx)

df_sorted = df.sort_values(by='mse', ascending=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# Print the sorted DataFrame
print(df_sorted[[
    'config/lr',
    'config/epochs',
    'config/batch_size',
    'config/win',
    'config/layers',
    'mse']])

print(result_grid.get_best_result(metric='mse', mode='min'))
