import numpy as np
from sklearn.metrics import mean_squared_error
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import timeseries_train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocess import preprocess_data, prep_data

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
for worm_num in range(5):
    print(worm_num)
    algorithm = 'rnn_autoencoder'
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
    _, x = preprocess_data(x, float(data.fps))
    x_, b_ = prep_data(x, b, win=config['win'])

    # train test split
    x_train, x_test, b_train, b_test = timeseries_train_test_split(x_, b_)
    x0_tr = x_train[:, 0, :, :]
    x0_tst = x_test[:, 0, :, :]
    x1_tr = x_train[:, 1, :, :]
    x1_tst = x_test[:, 1, :, :]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    model = RnnAutoencoder(input_dim=x0_tr.shape[2], hidden_dim=config['hidden_dim'] ,latent_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()

    x0_tr_tensor = torch.tensor(x0_tr, dtype=torch.float32).to(device)
    x0_tst_tensor = torch.tensor(x0_tst, dtype=torch.float32).to(device)
    x1_tr_tensor = torch.tensor(x1_tr, dtype=torch.float32).to(device)
    x1_tst_tensor = torch.tensor(x1_tst, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(x0_tr_tensor), batch_size=config['batch_size'], shuffle=True)
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
        reconstructed, _ = model(x0_tr_tensor)
        print(reconstructed.size())
        loss = mean_squared_error(
            x0_tr_tensor.view(x0_tr_tensor.size(0), -1).cpu().numpy(),
            reconstructed.view(reconstructed.size(0), -1).cpu().numpy()
        )
        print('mse:', round(loss, 8))

    # project into latent space
    with torch.no_grad():
        _, y0_tr = model(x0_tr_tensor)
        _, y0_tst = model(x0_tst_tensor)
        _, y1_tr = model(x1_tr_tensor)
        _, y1_tst = model(x1_tst_tensor)
        y0_tr, y0_tst, y1_tr, y1_tst = (
            y0_tr.cpu().numpy(),
            y0_tst.cpu().numpy(),
            y1_tr.cpu().numpy(),
            y1_tst.cpu().numpy(),
        )

    # saving
    save_model = True
    if save_model:
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y0_tr__{algorithm}_worm_{worm_num}', y0_tr)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y1_tr__{algorithm}_worm_{worm_num}', y1_tr)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y0_tst__{algorithm}_worm_{worm_num}', y0_tst)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y1_tst__{algorithm}_worm_{worm_num}', y1_tst)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/b_tr__{algorithm}_worm_{worm_num}', b_train)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/b_tst__{algorithm}_worm_{worm_num}', b_test)
