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
algorithm = 'rnn_autoencoder'
b_neurons = ['AVAR', 'AVAL', 'SMDVR', 'SMDVL', 'SMDDR', 'SMDDL', 'RIBR', 'RIBL']
data_path = 'data/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
data.exclude_neurons(b_neurons)
x = data.neuron_traces.T
b = data.behaviour

# hyperparameter tuning function
def train_rnn_autoencoder(config):
    # prepare data with given  dow size
    x_, b_ = prep_data(x, b, win=int(config["win"]))
    x0_ = x_[:, 0, :, :]
    x0_ = torch.tensor(x0_, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(x0_), batch_size=int(config["batch_size"]), shuffle=True)

    # RNN autoencoder model
    class RnnAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super(RnnAutoencoder, self).__init__()
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim

            # RNN Encoder
            self.encoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.to_embedding = nn.Linear(hidden_dim, latent_dim)
            self.from_embedding = nn.Linear(latent_dim, hidden_dim)
            self.decoder_rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            self.readout = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            batch_size, seq_len, input_dim = x.size()
            _, h_n = self.encoder_rnn(x)
            embedding = self.to_embedding(h_n.squeeze(0))
            h0 = self.from_embedding(embedding).unsqueeze(0)
            dummy_input = torch.zeros((batch_size, seq_len, input_dim), device=x.device)
            decoded_seq, _ = self.decoder_rnn(dummy_input, h0)
            output = self.readout(decoded_seq)
            return output, embedding

    # model initialization
    model = RnnAutoencoder(input_dim=x0_.shape[2], hidden_dim=int(config["hidden_dim"]), latent_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()

    # training loop
    for epoch in range(int(config["epochs"])):
        model.train()
        for batch in train_loader:
            x_batch = batch[0]
            optimizer.zero_grad()
            reconstructed, _ = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()

    # evaluation
    model.eval()
    with torch.no_grad():
        reconstructed, _ = model(x0_)
        model_mse = mean_squared_error(x0_.view(x0_.size(0), -1).numpy(), reconstructed.view(reconstructed.size(0), -1).numpy())

    # report result to ray tune
    tune.report({"mse": model_mse})


if __name__ == "__main__":

    max_epochs = 2000
    # Hyperparameter search space
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "epochs": tune.loguniform(10, max_epochs),
        "batch_size": tune.qloguniform(32, 512, 16),
        "win": tune.loguniform(1, 20),
        "hidden_dim": tune.qloguniform(32, 256, 16)
    }

    # hyperparameter tuning
    search_algo = BayesOptSearch(metric="mse", mode="min")

    tuner = tune.Tuner(
        tune.with_parameters(train_rnn_autoencoder),
        tune_config=tune.TuneConfig(
            search_alg=search_algo,
            num_samples=200,
            max_concurrent_trials=7
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric='mse', mode='min')
    print("Minimum validation loss:", best_result.metrics['mse'])
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

