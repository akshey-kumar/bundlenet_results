import sys
sys.path.append(r'../')
import numpy as np
from functions import *

algorithm = 'autoencoder'

for worm_num in range(5):
	### Load Data (and excluding behavioural neurons)
	b_neurons = [
		'AVAR',
		'AVAL',
		'SMDVR',
		'SMDVL',
		'SMDDR',
		'SMDDL',
		'RIBR',
		'RIBL',]
	data = Database(data_set_no=worm_num)
	data.exclude_neurons(b_neurons)
	X = data.neuron_traces.T
	B = data.states
	state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']

	### Preprocess and prepare data for BundLe Net
	time, X = preprocess_data(X, data.fps)
	X_, B_ = prep_data(X, B, win=1)

	## Train test split 
	X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)
	X0_tr = X_train[:,0,:,:]
	X1_tr = X_train[:,1,:,:]
	X0_tst = X_test[:,0,:,:]
	X1_tst = X_test[:,1,:,:]

	### Autoencoder architecture
	class Autoencoder(Model):
		def __init__(self, latent_dim):
			super(Autoencoder, self).__init__()
			self.latent_dim = latent_dim   
			self.encoder = tf.keras.Sequential([
					layers.Flatten(),
					layers.Dense(50, activation='relu'),
					layers.Dense(30, activation='relu'),
					layers.Dense(25, activation='relu'),
					layers.Dense(10, activation='relu'),
					layers.Dense(latent_dim, activation='linear'),
			])
			self.decoder = tf.keras.Sequential([
					layers.Dense(latent_dim, activation='relu'),
					layers.Dense(10, activation='relu'),
					layers.Dense(25, activation='relu'),
					layers.Dense(30, activation='relu'),
					layers.Dense(50, activation='relu'),
					layers.Dense(X0_tr.shape[-1]*X0_tr.shape[-2], activation='linear'),
					layers.Reshape(X0_tr.shape[1:])
			])
		def call(self, x):
			encoded = self.encoder(x)
			decoded = self.decoder(encoded)
			return decoded

	### Deploy Autoencoder
	autoencoder = Autoencoder(latent_dim = 3)
	opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
	autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse'])
	
	history = autoencoder.fit(X0_tr,
												X0_tr,
												epochs=150,
												batch_size=100,
												validation_data=(X0_tst, X0_tst),
												verbose=False
												)

	X0_pred = autoencoder(X0_tst).numpy()
	modelmse_tst = mean_squared_error(flat_partial(X0_tst), flat_partial(X0_pred))
	print('Test set mse:', modelmse_tst.round(8))

	### Projecting into latent space
	Y0_tr = autoencoder.encoder(X0_tr).numpy()
	Y1_tr = autoencoder.encoder(X1_tr).numpy()

	Y0_tst = autoencoder.encoder(X0_tst).numpy()
	Y1_tst = autoencoder.encoder(X1_tst).numpy()

	# Save the weights
	# model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
	np.savetxt('data/generated/saved_Y/Y0_tr__'+algorithm+'_worm_'+ str(worm_num), Y0_tr)
	np.savetxt('data/generated/saved_Y/Y1_tr__'+algorithm+'_worm_'+ str(worm_num), Y1_tr)
	np.savetxt('data/generated/saved_Y/Y0_tst__'+algorithm+'_worm_'+ str(worm_num), Y0_tst)
	np.savetxt('data/generated/saved_Y/Y1_tst__'+algorithm+'_worm_'+ str(worm_num), Y1_tst)
	np.savetxt('data/generated/saved_Y/B_train_1__'+algorithm+'_worm_'+ str(worm_num), B_train_1)
	np.savetxt('data/generated/saved_Y/B_test_1__'+algorithm+'_worm_'+ str(worm_num), B_test_1)



