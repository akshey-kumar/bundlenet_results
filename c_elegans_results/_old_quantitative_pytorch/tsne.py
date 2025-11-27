import os
import numpy as np
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import timeseries_train_test_split
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from sklearn.manifold import TSNE
from preprocess import preprocess_data, prep_data

# load data (excluding behavioural neurons) and plot
for worm_num in range(5):
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
    '''
    Best hyperparameters found were:
    perplexity: 11.85014200446232
    early_exaggeration: 12.163332530023283
    max_iter: 2349.109910160068
    win: 16.85702251549715 round --> 17
    '''
    # prepare data
    _, x = preprocess_data(x, float(data.fps))
    x_, b_ = prep_data(x, b, win=17)

    ## train test split
    x_train, x_test, b_train, b_test = timeseries_train_test_split(x_, b_)
    x0_tr = x_train[:, 0, :, :].reshape(x_train.shape[0], -1)
    x1_tr = x_train[:, 1, :, :].reshape(x_train.shape[0], -1)
    x0_tst = x_test[:, 0, :, :].reshape(x_test.shape[0], -1)
    x1_tst = x_test[:, 1, :, :].reshape(x_test.shape[0], -1)

    # deploy tsne
    dim = 3
    tsne = TSNE(n_components=dim, init='pca', perplexity=80)

    # projecting into latent space
    y0_tr = tsne.fit_transform(x0_tr)
    y1_tr = tsne.fit_transform(x1_tr)

    '''tsne does not have transform method alone only fittransform. 
    Hence it cannot be run separately on held out test data. So we
    set the test data equal to the train data'''
    y0_tst = y0_tr.copy()
    y1_tst = y1_tr.copy()
    b_test = b_train.copy()

    # save the weights
    save_model = True
    if save_model:
        os.makedirs('data/generated/quantitative_evaluation/embeddings/c_elegans/', exist_ok=True)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y0_tr__{algorithm}_worm_{worm_num}', y0_tr)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y1_tr__{algorithm}_worm_{worm_num}', y1_tr)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y0_tst__{algorithm}_worm_{worm_num}', y0_tst)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y1_tst__{algorithm}_worm_{worm_num}', y1_tst)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/b_tr__{algorithm}_worm_{worm_num}', b_train)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/b_tst__{algorithm}_worm_{worm_num}', b_test)