import numpy as np
from sklearn.metrics import mean_squared_error
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Best hyperparameters found from tuning were:
config = {
    'lr': 0.0005096067531892613 ,
    'epochs': int(1614.8131428890506),
    'batch_size': int(52.136172155693295),
    'win': int(5.341348373587461),
    'hidden_dim': int(167.28320332656327),
}

class RnnAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RnnAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # RNN Encoder
        self.encoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # Map final hidden state to low-dimensional embedding
        self.to_embedding = nn.Linear(hidden_dim, latent_dim)
        # Map embedding back to generator initial state
        self.from_embedding = nn.Linear(latent_dim, hidden_dim)
        # Generator RNN (decoder)
        self.decoder_rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        # Readout layer to project hidden state to original neuron space
        self.readout = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, window_len, input_dim = x.size()

        # --- Encode ---
        _, h_n = self.encoder_rnn(x)  # h_n shape: (1, batch, hidden_dim)
        embedding = self.to_embedding(h_n.squeeze(0))  # (batch, latent_dim)
        # --- Decode ---
        h0 = self.from_embedding(embedding).unsqueeze(0)  # (1, batch, hidden_dim)
        dummy_input = torch.zeros((batch_size, window_len, input_dim), device=x.device)
        decoded_seq, _ = self.decoder_rnn(dummy_input, h0)  # (batch, time, hidden_dim)
        output = self.readout(decoded_seq)  # (batch, time, input_dim)

        return output, embedding


# Training and evaluation
for worm_num in range(1,5):
    print(worm_num)
    algorithm = 'rnn_autoencoder'
    b_neurons = ['AVAR', 'AVAL', 'SMDVR', 'SMDVL', 'SMDDR', 'SMDDL', 'RIBR', 'RIBL']
    data_path = 'data/raw/c_elegans/NoStim_Data.mat'
    data = Database(data_path=data_path, dataset_no=worm_num)
    data.exclude_neurons(b_neurons)
    x = data.neuron_traces.T
    b = data.behaviour

    # prepare data
    x_, b_ = prep_data(x, b, win=config['win'])
    x0_ = x_[:, 0, :, :]  # (n_windows, window_len, n_neurons)

    # five fits of the model and pick the best model
    best_model = None
    lowest_loss = float("inf")
    for _ in range(5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', device)
        model = RnnAutoencoder(input_dim=x0_.shape[2], hidden_dim=config['hidden_dim'] ,latent_dim=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.MSELoss()
        x0_tensor = torch.tensor(x0_, dtype=torch.float32).to(device)
        train_loader = DataLoader(TensorDataset(x0_tensor), batch_size=config['batch_size'], shuffle=True)
        epochs = config['epochs']
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                x_batch = batch[0].to(device)
                optimizer.zero_grad()
                reconstructed, _ = model(x_batch)
                loss = criterion(reconstructed, x_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            reconstructed, _ = model(x0_tensor)
            loss = mean_squared_error(
                x0_tensor.view(x0_tensor.size(0), -1).cpu().numpy(),
                reconstructed.view(reconstructed.size(0), -1).cpu().numpy()
            )
            print('mse:', round(loss, 8))

        if loss < lowest_loss:
            best_model, lowest_loss = model, loss

    # project into latent space
    with torch.no_grad():
        _, y0_ = best_model(x0_tensor)
        y0_ = y0_.cpu().numpy()

    # plotting latent space dynamics
    # vis = LatentSpaceVisualiser(y0_, b_, data.behaviour_names)
    # vis.plot_latent_timeseries()
    # vis.plot_phase_space()

    save_model = True
    if save_model:
        np.savetxt(f'data/generated/embeddings/c_elegans/y0__{algorithm}_worm_{worm_num}', y0_)
        np.savetxt(f'data/generated/embeddings/c_elegans/b__{algorithm}_worm_{worm_num}', b_)
        y0_ = np.loadtxt(f'data/generated/embeddings/c_elegans/y0__{algorithm}_worm_{worm_num}')
        b_ = np.loadtxt(f'data/generated/embeddings/c_elegans/b__{algorithm}_worm_{worm_num}').astype(int)

