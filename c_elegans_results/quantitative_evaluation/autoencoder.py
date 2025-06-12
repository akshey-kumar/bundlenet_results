import numpy as np
from sklearn.metrics import mean_squared_error
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# load data (excluding behavioural neurons) and plot
for worm_num in range(5):
    algorithm = 'autoencoder'
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
    '''
    Best hyperparameters found were:
    lr: 0.0010628744197015334
    epochs: 267.8271592227043 int --> 267
    batch_size: 276.42401266792365 int --> 276
    win: 10.97115164072503  int --> 10
    layers_idx: 1.6918718067487664 --> [100, 150, 50, 10],  # Increasing then Decreasing Architecture
    '''
    # prepare data (This autoencoder predicts the difference between present and future state)
    x_, b_ = prep_data(x, b, win=10)

    # train test split
    x_train, x_test, b_train, b_test = timeseries_train_test_split(x_, b_)
    x0_tr = x_train[:, 0, :, :]
    x1_tr = x_train[:, 1, :, :]
    xdiff_tr = x1_tr - x0_tr
    x0_tst = x_test[:, 0, :, :]
    x1_tst = x_test[:, 1, :, :]
    xdiff_tst = x1_tst - x0_tst

    # scaling input and output data
    xdmax = (np.abs(xdiff_tr)).max()  # Parameters for scaling
    xdiff_tr, xdiff_tst = xdiff_tr / xdmax, xdiff_tst / xdmax

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

    # fit the autoencoder to data
    latent_dim = 3
    model = Autoencoder(latent_dim, input_shape=x0_tr.shape)
    optimizer = optim.Adam(model.parameters(), lr=0.0010628744197015334)
    criterion = nn.MSELoss()
    x0_tr = torch.tensor(x0_tr, dtype=torch.float32)
    x1_tr = torch.tensor(x1_tr, dtype=torch.float32)
    x0_tst = torch.tensor(x0_tst, dtype=torch.float32)
    x1_tst = torch.tensor(x1_tst, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(x0_tr, x0_tr), batch_size=276, shuffle=True)

    epochs = 267
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
        x0_tr_pred = model(x0_tr).numpy()
        x0_tst_pred = model(x0_tst).numpy()

    loss_tr = mean_squared_error(x0_tr.view(x0_tr.size(0), -1).numpy(), x0_tr_pred.reshape(x0_tr_pred.shape[0], -1))
    loss_tst = mean_squared_error(x0_tst.view(x0_tst.size(0), -1).numpy(), x0_tst_pred.reshape(x0_tst_pred.shape[0], -1))

    print('train mse:', round(loss_tr, 8))
    print('test mse:', round(loss_tst, 8))

    # project into latent space
    with torch.no_grad():
        y0_tr = model.encoder(x0_tr).numpy()
        y1_tr = model.encoder(x1_tr).numpy()
        y0_tst = model.encoder(x0_tst).numpy()
        y1_tst = model.encoder(x1_tst).numpy()

    # saving
    save_model = True
    if save_model:
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y0_tr__{algorithm}_worm_{worm_num}', y0_tr)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y1_tr__{algorithm}_worm_{worm_num}', y1_tr)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y0_tst__{algorithm}_worm_{worm_num}', y0_tst)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y1_tst__{algorithm}_worm_{worm_num}', y1_tst)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/b_tr__{algorithm}_worm_{worm_num}', b_train)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/b_tst__{algorithm}_worm_{worm_num}', b_test)
