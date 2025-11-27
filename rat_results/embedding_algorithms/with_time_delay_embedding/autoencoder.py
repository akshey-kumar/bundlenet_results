import numpy as np
from sklearn.metrics import mean_squared_error
from ncmcm.bundlenet.utils import prep_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


algorithm = 'autoencoder'
# optimal hyperparameters chosen as a result of parameter tuning
# see rat_results/embedding_algorithms/hyperparameter_optimisation/autoencoder.py
'''
Best hyperparameters found were:
lr: 0.015540653524918275
epochs: 401
batch_size: 511
win: 1
layers_idx: 1 -> [100, 150, 50, 10],  # Increasing then Decreasing Architecture
'''
for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    # Load data
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values

    # time delay embedding
    x_, b_ = prep_data(x, b, win=1)
    x_ = x_[:, -1, :, :]

    # autoencoder architecture
    class Autoencoder(nn.Module):
        def __init__(self, latent_dim, input_shape):
            super(Autoencoder, self).__init__()
            self.latent_dim = latent_dim
            in_features = np.prod(input_shape[-2:])  # Flattened input size

            # Encoder
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 100),
                nn.ReLU(),
                nn.Linear(100, 150),
                nn.ReLU(),
                nn.Linear(150, 50),
                nn.ReLU(),
                nn.Linear(50, 10),
                nn.ReLU(),
                nn.Linear(10, latent_dim)  # Linear activation (default)
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 10),
                nn.ReLU(),
                nn.Linear(10, 50),
                nn.ReLU(),
                nn.Linear(50, 150),
                nn.ReLU(),
                nn.Linear(150, 100),
                nn.ReLU(),
                nn.Linear(100, in_features),
                nn.Linear(in_features, in_features),  # Linear activation (default)
                nn.Unflatten(1, input_shape[-2:])  # Reshape back to original shape
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


    # five fits of the model and pick the best model
    best_model = None
    lowest_loss = float("inf")
    for _ in range(5):
        # fit the autoencoder to data
        latent_dim = 3
        model = Autoencoder(latent_dim, input_shape=x_.shape)
        optimizer = optim.Adam(model.parameters(), lr=0.015540653524918275)
        criterion = nn.MSELoss()
        x_ = torch.tensor(x_, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(x_, x_), batch_size=511, shuffle=True)

        epochs = 401
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                x_batch, _ = batch
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, x_batch)
                loss.backward()
                optimizer.step()

        # evaluate
        model.eval()
        with torch.no_grad():
            x_pred = model(x_).numpy()

        loss = mean_squared_error(x_.numpy().reshape(x_.shape[0], -1), x_pred.reshape(x_pred.shape[0], -1))
        print('mse:', round(loss, 8))

        if loss < lowest_loss:
            best_model, lowest_loss = model, loss

    # project into latent space
    with torch.no_grad():
        y_ = best_model.encoder(x_).numpy()

    # save the weights
    save_model = True
    if save_model:
        np.savetxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}', y_)
        np.savetxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}', b_)
        y_ = np.loadtxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}')
        b_ = np.loadtxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}').astype(int)



