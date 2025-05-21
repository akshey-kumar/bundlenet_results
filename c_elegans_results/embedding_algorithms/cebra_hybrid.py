import numpy as np
# from sklearn.decomposition import PCA
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, model_inference
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from cebra import CEBRA

# load Data (excluding behavioural neurons) and plot
worm_num = 0
algorithm = 'cebra_hybrid'
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

# prepare data
x_, b_ = prep_data(x, b, win=1)

'''
From hyperparameter optimisation we got:

Minimum validation loss: 0.09375
Best hyperparameters found were:  {'model_architecture': 'offset1-model-mse', 'batch_size': 512, 'learning_rate': 1.7239766706958178e-05, 'temperature': 0.1870932548798307, 'max_iterations': 10000, 'distance': 'cosine', 'time_offsets': 20}

'''
# fit CEBRA hybrid
cebra_hybrid_model = CEBRA(
	model_architecture='offset10-model',
	batch_size=512,
	learning_rate=3e-4,
	temperature=1,
	output_dimension=3,
	max_iterations=5000,
	distance='cosine',
	conditional='time_delta',
	device='cuda_if_available',
	verbose=True,
	time_offsets=10,
	hybrid = True
)

cebra_hybrid_model.fit(x_[:, 0, 0, :], b_.astype(float))

# projecting into latent space
y0_ = cebra_hybrid_model.transform(x_[:, 0, 0, :])

# save the weights
save_model = False
if save_model:
    #model.save_weights(f'data/generated/models/bunDLeNet_model_worm_{worm_num}')
    np.savetxt(f'data/generated/embeddings/y0__{algorithm}_worm_{worm_num}', y0_)
    np.savetxt(f'data/generated/embeddings/b__{algorithm}_worm_{worm_num}', b_)
    y0_ = np.loadtxt(f'data/generated/embeddings/y0__{algorithm}_worm_{worm_num}')
    b_ = np.loadtxt(f'data/generated/embeddings/b__{algorithm}_worm_{worm_num}').astype(int)

# plotting latent space dynamics
vis = LatentSpaceVisualiser(y0_, b_, data.behaviour_names)
vis.plot_latent_timeseries()
vis.plot_phase_space()
#vis.rotating_plot(filename='figures/rotation_' + algorithm + '_worm_' + str(worm_num) + '.gif')
