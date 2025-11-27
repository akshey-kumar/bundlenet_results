import sys
sys.path.append(r'../')
import numpy as np
from functions import *

algorithm = 'tsne'

### Load Data (and excluding behavioural neurons)
for worm_num in range(5):
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

	### Deploy tsne
	dim = 3
	tsne = TSNE(n_components=dim, init='pca', perplexity=80)
	
	### Projecting into latent space
	Y0_tr = tsne.fit_transform(X_train[:,0,0,:])
	Y1_tr = tsne.fit_transform(X_train[:,1,0,:])

	'''tsne does not have transform method alone only fittransform. 
	Hence it cannot be run separately on held out test data. So we
	set the test data equal to the train data'''
	Y0_tst = Y0_tr.copy()
	Y1_tst = Y1_tr.copy()
	B_test_1 = B_train_1.copy()

	# Save the weights
	# model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
	np.savetxt('data/generated/saved_Y/Y0_tr__'+algorithm+'_worm_'+ str(worm_num), Y0_tr)
	np.savetxt('data/generated/saved_Y/Y1_tr__'+algorithm+'_worm_'+ str(worm_num), Y1_tr)
	np.savetxt('data/generated/saved_Y/Y0_tst__'+algorithm+'_worm_'+ str(worm_num), Y0_tst)
	np.savetxt('data/generated/saved_Y/Y1_tst__'+algorithm+'_worm_'+ str(worm_num), Y1_tst)
	np.savetxt('data/generated/saved_Y/B_train_1__'+algorithm+'_worm_'+ str(worm_num), B_train_1)
	np.savetxt('data/generated/saved_Y/B_test_1__'+algorithm+'_worm_'+ str(worm_num), B_test_1)
