import numpy as np
from sklearn.metrics import mean_squared_error
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# load data (excluding behavioural neurons) and plot
for worm_num in range(5):
    algorithm = 'autoregressor_autoencoder'
    b_neurons = [
        'AVAR',
        'AVAL',
        'SMDVR',
        'SMDVL',
        'SMDDR',
        'SMDDL',
        'RIBR',
        'RIBL'
    ]
    data_path = 'data/raw/c_elegans/NoStim_Data.mat'
    data = Database(data_path=data_path, dataset_no=worm_num)
    data.exclude_neurons(b_neurons)
    x = data.neuron_traces.T
    b = data.behaviour

    # prepare data (This autoencoder predicts the difference between present and future state)
    x_, b_ = prep_data(x, b, win=5)
    x0_ = x_[:, 0, :, :]
    x1_ = x_[:, 1, :, :]
    xdiff_ = x1_ - x0_

    # scaling input and output data
    xdmax = (np.abs(xdiff_)).max() # Parameters for scaling
    xdiff_ = xdiff_/xdmax


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
                nn.Linear(10, latent_dim)  # Linear activation (default)
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
                nn.Unflatten(1, input_shape[-2:])  # Reshape back to original shape
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


    # fit the autoencoder to data
    latent_dim = 3
    model = Autoencoder(latent_dim=3, input_shape=x0_.shape)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    x0_ = torch.tensor(x0_, dtype=torch.float32)
    xdiff_ = torch.tensor(xdiff_, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(x0_, xdiff_), batch_size=100, shuffle=True)

    epochs = 150
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
        xdiff_pred = model(x0_).numpy()
        # inverse scaling the data
        xdiff_pred, xdiff_ = xdiff_pred*xdmax, xdiff_*xdmax
        x1_pred = x0_.numpy() + xdiff_pred


    model_mse = mean_squared_error(x1_.reshape(x1_.shape[0], -1), x1_pred.reshape(x1_pred.shape[0], -1))
    print('mse:', round(model_mse, 8))

    # project into latent space
    with torch.no_grad():
        y0_ = model.encoder(x0_).numpy()

    # save the weights
    save_model = True
    if save_model:
        np.savetxt(f'data/generated/embeddings/y0__{algorithm}_worm_{worm_num}', y0_)
        np.savetxt(f'data/generated/embeddings/b__{algorithm}_worm_{worm_num}', b_)
        y0_ = np.loadtxt(f'data/generated/embeddings/y0__{algorithm}_worm_{worm_num}')
        b_ = np.loadtxt(f'data/generated/embeddings/b__{algorithm}_worm_{worm_num}').astype(int)

    # plotting latent space dynamics
    #vis = LatentSpaceVisualiser(y0_, b_, data.behaviour_names)
    #vis.plot_latent_timeseries()
    #vis.plot_phase_space()
