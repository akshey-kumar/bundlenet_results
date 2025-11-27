import sys
sys.path.append(r'../')
import numpy as np
from functions import *

algorithm = 'PCA'

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
	time, X = preprocess_data(X, float(data.fps))
	X_, B_ = prep_data(X, B, win=1)

	## Train test split 
	X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)

	### Deploy PCA
	dim = 3
	pca = PCA(n_components=dim)
	pca.fit(X_train[:,0,0,:])
	print('Percentage of variance explained by the first ', dim, ' PCs: ', pca.explained_variance_ratio_[:dim].sum().round(3))
	
	### Projecting into latent space
	Y0_tr = pca.transform(X_train[:,0,0,:])
	Y1_tr = pca.transform(X_train[:,1,0,:])

	Y0_tst = pca.transform(X_test[:,0,0,:])
	Y1_tst = pca.transform(X_test[:,1,0,:])

	# Save the weights
	# model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
	np.savetxt('data/generated/saved_Y/Y0_tr__'+algorithm+'_worm_'+ str(worm_num), Y0_tr)
	np.savetxt('data/generated/saved_Y/Y1_tr__'+algorithm+'_worm_'+ str(worm_num), Y1_tr)
	np.savetxt('data/generated/saved_Y/Y0_tst__'+algorithm+'_worm_'+ str(worm_num), Y0_tst)
	np.savetxt('data/generated/saved_Y/Y1_tst__'+algorithm+'_worm_'+ str(worm_num), Y1_tst)
	np.savetxt('data/generated/saved_Y/B_train_1__'+algorithm+'_worm_'+ str(worm_num), B_train_1)
	np.savetxt('data/generated/saved_Y/B_test_1__'+algorithm+'_worm_'+ str(worm_num), B_test_1)