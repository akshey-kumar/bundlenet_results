import sys
import numpy as np
from functions import Database, preprocess_data, prep_data, timeseries_train_test_split
from cebra import CEBRA

algorithm = 'cebra_h'
'''
Best hyperparameters found were:
model_architecture: offset1-model-mse
batch_size: 256
learning_rate: 0.00025862916104496357
temperature: 0.4571735639111726
max_iterations: 2000
distance: euclidean
time_offsets: 5
'''
config = {
	'model_architecture': 'offset1-model-mse',
	'batch_size': 256,
	'learning_rate': 0.00025862916104496357,
	'temperature': 0.4571735639111726,
	'max_iterations': 2000,
	'distance': 'euclidean',
	'time_offsets': 5,
}

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

	# fit CEBRA hybrid
	cebra_hybrid_model = CEBRA(
		model_architecture=config["model_architecture"],
		batch_size=config["batch_size"],
		learning_rate=config["learning_rate"],
		temperature=config["temperature"],
		output_dimension=3,
		max_iterations=config["max_iterations"],
		distance=config["distance"],
		conditional='time_delta',
		device='cuda_if_available',
		verbose=True,
		time_offsets=config["time_offsets"],
		hybrid=True
	)

	cebra_hybrid_model.fit(X_train[:,0,0,:], B_train_1.astype(float))
	print(worm_num)

	### Projecting into latent space
	Y0_tr = cebra_hybrid_model.transform(X_train[:,0,0,:])
	Y1_tr = cebra_hybrid_model.transform(X_train[:,1,0,:])

	Y0_tst = cebra_hybrid_model.transform(X_test[:,0,0,:])
	Y1_tst = cebra_hybrid_model.transform(X_test[:,1,0,:])

	# Save the weights
	# model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
	np.savetxt('data/generated/saved_Y/Y0_tr__'+algorithm+'_worm_'+ str(worm_num), Y0_tr)
	np.savetxt('data/generated/saved_Y/Y1_tr__'+algorithm+'_worm_'+ str(worm_num), Y1_tr)
	np.savetxt('data/generated/saved_Y/Y0_tst__'+algorithm+'_worm_'+ str(worm_num), Y0_tst)
	np.savetxt('data/generated/saved_Y/Y1_tst__'+algorithm+'_worm_'+ str(worm_num), Y1_tst)
	np.savetxt('data/generated/saved_Y/B_train_1__'+algorithm+'_worm_'+ str(worm_num), B_train_1)
	np.savetxt('data/generated/saved_Y/B_test_1__'+algorithm+'_worm_'+ str(worm_num), B_test_1)
	'''
	Y0_tr = np.loadtxt('data/generated/saved_Y/Y0_tr__'+algorithm+'_worm_'+ str(worm_num))
	Y1_tr = np.loadtxt('data/generated/saved_Y/Y1_tr__'+algorithm+'_worm_'+ str(worm_num))
	Y0_tst = np.loadtxt('data/generated/saved_Y/Y0_tst__'+algorithm+'_worm_'+ str(worm_num))
	Y1_tst = np.loadtxt('data/generated/saved_Y/Y1_tst__'+algorithm+'_worm_'+ str(worm_num))
	B_train_1 = np.loadtxt('data/generated/saved_Y/B_train_1__'+algorithm+'_worm_'+ str(worm_num)).astype(int)
	B_test_1 = np.loadtxt('data/generated/saved_Y/B_test_1__'+algorithm+'_worm_'+ str(worm_num)).astype(int)

	plot_phase_space(Y0_tr, B_train_1, state_names)
	plot_phase_space(Y0_tst, B_test_1, state_names)
	'''