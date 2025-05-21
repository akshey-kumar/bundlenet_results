import numpy as np
from sklearn.decomposition import PCA
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, model_inference
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

# load Data (excluding behavioural neurons) and plot
for worm_num in range(5):
    algorithm = 'pca'
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
    # time delay embedding
    x_ = x_[:,1,:,:].reshape(x_.shape[0], -1)

    # fit PCA
    dim = 3
    pca = PCA(n_components=dim)
    pca.fit(x_)
    print('Percentage of variance explained by the first ', dim, ' PCs: ', pca.explained_variance_ratio_[:dim].sum().round(3))

    # projecting into latent space
    y_ = pca.transform(x_)

    # save the weights
    save_model = True
    if save_model:
        #model.save_weights(f'data/generated/models/bunDLeNet_model_worm_{worm_num}')
        np.savetxt(f'data/generated/embeddings/y0__{algorithm}_worm_{worm_num}', y_)
        np.savetxt(f'data/generated/embeddings/b__{algorithm}_worm_{worm_num}', b_)
        y_ = np.loadtxt(f'data/generated/embeddings/y0__{algorithm}_worm_{worm_num}')
        b_ = np.loadtxt(f'data/generated/embeddings/b__{algorithm}_worm_{worm_num}').astype(int)

    # plotting latent space dynamics
    # vis = LatentSpaceVisualiser(y_, b_, data.behaviour_names)
    # vis.plot_latent_timeseries()
    # vis.plot_phase_space()
