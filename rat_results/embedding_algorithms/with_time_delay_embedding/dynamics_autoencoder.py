
import numpy as np
from sklearn.metrics import mean_squared_error
from ncmcm.bundlenet.utils import prep_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


algorithm = 'dynamics_autoencoder'
# optimal hyperparameters chosen as a result of parameter tuning
# see rat_results/embedding_algorithms/hyperparameter_optimisation/dynamics_autoencoder.py
'''
Best hyperparameters found were:
lr: 0.004169807344148425
epochs: 992
batch_size: 511
win: 2
layers_idx: 2 -> [50, 30, 25, 10]
'''
for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    # Load data
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values

    # time delay embedding
    x_, b_ = prep_data(x, b, win=2)
    x0_ = x_[:, 0, :, :]
    x1_ = x_[:, 1, :, :]
    xdiff_ = x1_ - x0_

    # scaling input and output data
    xdmax = (np.abs(xdiff_)).max()  # Parameters for scaling
    xdiff_ = xdiff_ / xdmax

    # autoencoder architecture
    class Autoencoder(nn.Module):
        def __init__(self, latent_dim, input_shape):
            super(Autoencoder, self).__init__()
            self.latent_dim = latent_dim
            in_features = np.prod(input_shape[-2:])  # Flattened input size

            # Encoder
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 50),
                nn.ReLU(),
                nn.Linear(50, 30),
                nn.ReLU(),
                nn.Linear(30, 25),
                nn.ReLU(),
                nn.Linear(25, 10),
                nn.ReLU(),
                nn.Linear(10, latent_dim)
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 10),
                nn.ReLU(),
                nn.Linear(10, 25),
                nn.ReLU(),
                nn.Linear(25, 30),
                nn.ReLU(),
                nn.Linear(30, 50),
                nn.ReLU(),
                nn.Linear(50, in_features),
                nn.Linear(in_features, in_features),
                nn.Unflatten(1, input_shape[-2:])
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


    # data loader for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x0_tensor = torch.from_numpy(x0_).float().to(device)
    xdiff_tensor = torch.from_numpy(xdiff_).float().to(device)
    train_loader = DataLoader(TensorDataset(x0_tensor, xdiff_tensor), batch_size=511, shuffle=True)

    # five fits of the model and pick the best model
    best_model = None
    lowest_loss = float("inf")
    for _ in range(5):
        # fit the autoencoder to data
        model = Autoencoder(latent_dim=3, input_shape=x0_.shape).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.004169807344148425)
        criterion = nn.MSELoss()

        epochs = 992
        for epoch in range(epochs):
            model.train()
            for x_batch, xdiff_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, xdiff_batch)
                loss.backward()
                optimizer.step()

        # evaluate
        model.eval()
        with torch.no_grad():
            xdiff_pred = model(x0_tensor).cpu().numpy()
            # inverse scaling the data
            xdiff_pred, xdiff_ = xdiff_pred * xdmax, xdiff_ * xdmax
            x1_pred = x0_ + xdiff_pred

        loss = mean_squared_error(x1_.reshape(x1_.shape[0], -1), x1_pred.reshape(x1_pred.shape[0], -1))
        print('mse:', round(loss, 8))

        if loss < lowest_loss:
            best_model, lowest_loss = model, loss

    print('lowest loss achieved:', round(lowest_loss, 8))
    # project into latent space
    with torch.no_grad():
        y_ = best_model.encoder(x0_tensor).cpu().numpy()

    # save the weights
    save_model = True
    if save_model:
        np.savetxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}', y_)
        np.savetxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}', b_)
        y_ = np.loadtxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}')
        b_ = np.loadtxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}').astype(int)
