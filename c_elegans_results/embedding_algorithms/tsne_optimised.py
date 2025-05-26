import numpy as np
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from sklearn.manifold import TSNE

# load data (excluding behavioural neurons) and plot
for worm_num in range(5):
    algorithm = 'tsne_optimised'
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
    perplexity: 11.85014200446232
    early_exaggeration: 12.163332530023283
    max_iter: 2349.109910160068
    win: 16.85702251549715 round --> 17
    '''
    # prepare data
    x_, b_ = prep_data(x, b, win=17)
    dim = 3

    # five fits of the model and pick the best model
    best_model = None
    lowest_loss = float("inf")
    for _ in range(5):
        tsne = TSNE(
            n_components=dim,
            init='pca',
            early_exaggeration=12.163332530023283,
            perplexity=11.85014200446232,
            max_iter=2349
        )
        # projecting into latent space
        embedding = tsne.fit_transform(x_[:, 0, 0, :])

        loss = tsne.kl_divergence_
        if loss < lowest_loss:
            y0_, lowest_loss = embedding, loss


    # save the weights
    save_model = True
    if save_model:
        np.savetxt(f'data/generated/embeddings/y0__{algorithm}_worm_{worm_num}', y0_)
        np.savetxt(f'data/generated/embeddings/b__{algorithm}_worm_{worm_num}', b_)
        y0_ = np.loadtxt(f'data/generated/embeddings/y0__{algorithm}_worm_{worm_num}')
        b_ = np.loadtxt(f'data/generated/embeddings/b__{algorithm}_worm_{worm_num}').astype(int)
