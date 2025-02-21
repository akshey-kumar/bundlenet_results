import numpy as np
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from sklearn.manifold import TSNE

# load data (excluding behavioural neurons) and plot
worm_num = 0
algorithm = 'tsne'
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

# fit tsne
dim = 3
tsne = TSNE(n_components=dim, init='pca', perplexity=80)

# projecting into latent space
y0_ = tsne.fit_transform(x_[:, 0, 0, :])

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
