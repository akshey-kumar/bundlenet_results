import os
import numpy as np
from sklearn.decomposition import PCA
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split

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

    ## train test split
    x_train, x_test, b_train, b_test = timeseries_train_test_split(x_, b_)
    x0_tr = x_train[:, 0, :, :].reshape(x_train.shape[0], -1)
    x1_tr = x_train[:, 1, :, :].reshape(x_train.shape[0], -1)
    x0_tst = x_test[:, 0, :, :].reshape(x_test.shape[0], -1)
    x1_tst = x_test[:, 1, :, :].reshape(x_test.shape[0], -1)

    ### Deploy PCA
    dim = 3
    pca = PCA(n_components=dim)
    pca.fit(x0_tr)
    print('Percentage of variance explained by the first ', dim, ' PCs: ',
          pca.explained_variance_ratio_[:dim].sum().round(3))

    ### Projecting into latent space
    y0_tr = pca.transform(x0_tr)
    y1_tr = pca.transform(x1_tr)
    y0_tst = pca.transform(x0_tst)
    y1_tst = pca.transform(x1_tst)


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