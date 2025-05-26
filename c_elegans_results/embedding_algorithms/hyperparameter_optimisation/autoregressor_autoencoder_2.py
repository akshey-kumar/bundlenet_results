import os
import numpy as np
from sklearn.metrics import mean_squared_error
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import prep_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch


# load data
worm_num = 0
algorithm = 'autoencoder_autoregressor_2'
b_neurons = ['AVAR', 'AVAL', 'SMDVR', 'SMDVL', 'SMDDR', 'SMDDL', 'RIBR', 'RIBL']
data_path = 'data/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
data.exclude_neurons(b_neurons)
x = data.neuron_traces.T
b = data.behaviour

# hyperparameter tuning function
def train_autoencoder(config):
    # prepare data with given window size
    x_, b_ = prep_data(x, b, win=int(config["win"]))
    x0_ = x_[:, 0, :, :]
    x1_ = x_[:, 1, :, :]
    x0_ = torch.tensor(x0_, dtype=torch.float32)
    x1_ = torch.tensor(x1_, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(x0_, x1_), batch_size=int(config["batch_size"]), shuffle=True)

    # autoencoder architecture
    architectures = [
        [50, 10],  # Shallow Architecture
        [100, 150, 50, 10],  # Increasing then Decreasing Architecture
        [50, 30, 25, 10],  # Same encoder as in BunDLeNet
        [100, 80, 60, 40, 20, 10, 10],  # Deep Architecture
        [75, 25, 10]
    ]
    layers_idx = int(config["layers_idx"])
    layers = architectures[layers_idx]
    print(layers)

    class Autoencoder(nn.Module):
        def __init__(self, latent_dim, input_shape, layers):
            super(Autoencoder, self).__init__()
            in_features = np.prod(input_shape[-2:])

            encoder_layers = [nn.Flatten()]
            for units in layers:
                encoder_layers.append(nn.Linear(in_features, units))
                encoder_layers.append(nn.ReLU())
                in_features = units
            encoder_layers.append(nn.Linear(in_features, latent_dim))

            decoder_layers = [nn.Linear(latent_dim, layers[-1]), nn.ReLU()]
            for units in reversed(layers[:-1]):
                decoder_layers.append(nn.Linear(decoder_layers[-2].out_features, units))
                decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(decoder_layers[-2].out_features, np.prod(input_shape[-2:])))
            decoder_layers.append(nn.Unflatten(1, input_shape[-2:]))

            self.encoder = nn.Sequential(*encoder_layers)
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    # model initialization
    model = Autoencoder(latent_dim=3, input_shape=x0_.shape, layers=layers)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()

    # training loop
    for epoch in range(int(config["epochs"])):
        model.train()
        for x0_batch, x1_batch in train_loader:
            optimizer.zero_grad()
            output = model(x0_batch)
            loss = criterion(output, x1_batch)
            loss.backward()
            optimizer.step()

    # evaluation
    model.eval()
    with torch.no_grad():
        x1_pred = model(x0_).numpy()
    model_mse = mean_squared_error(x1_.reshape(x1_.shape[0], -1), x1_pred.reshape(x1_pred.shape[0], -1))

    # report result to ray tune
    tune.report({"mse":model_mse})


if __name__ == "__main__":

    # Hyperparameter search space
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "epochs": tune.loguniform(10, 1000),
        "batch_size": tune.qloguniform(32, 512, 16),
        "win": tune.loguniform(1, 20),
        "layers_idx": tune.uniform(0, 4)
    }

    # hyperparameter tuning
    search_algo = BayesOptSearch(metric="mse", mode="min")

    tuner = tune.Tuner(
        tune.with_parameters(train_autoencoder),
        tune_config=tune.TuneConfig(
            search_alg=search_algo,
            num_samples=200,
            max_concurrent_trials=7
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric='mse', mode='min')
    print("Minimum loss:", best_result.metrics['mse'])
    print("Best hyperparameters found were: ", best_result.config)

    # save best hyperparameters to a file
    results_dir = 'data/generated/optimal_hyperparameters/'
    os.makedirs(results_dir, exist_ok=True)

    best_params_path = os.path.join(results_dir, f"best_params_c_elegans_{worm_num}_{algorithm}.txt")
    with open(best_params_path, 'w') as f:
        f.write(f"Minimum mse: {best_result.metrics['mse']}\n")
        f.write("Best hyperparameters found were:\n")
        for param, value in best_result.config.items():
            f.write(f"{param}: {value}\n")

