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
	algorithm = 'autoencoder_optimised'
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
	# prepare data
	x_, b_ = prep_data(x, b, win=10)
	x0_ = x_[:, 0, :, :]

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
		model = Autoencoder(latent_dim, input_shape=x0_.shape)
		optimizer = optim.Adam(model.parameters(), lr=0.0010628744197015334)
		criterion = nn.MSELoss()
		x0_ = torch.tensor(x0_, dtype=torch.float32)
		train_loader = DataLoader(TensorDataset(x0_, x0_), batch_size=276, shuffle=True)

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
			x0_pred = model(x0_).numpy()

		loss = mean_squared_error(x0_.view(x0_.size(0), -1).numpy(), x0_pred.reshape(x0_pred.shape[0], -1))
		print('mse:', round(loss, 8))

		if loss < lowest_loss:
			best_model, lowest_loss = model, loss


	# project into latent space
	with torch.no_grad():
		y0_ = best_model.encoder(x0_).numpy()

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

