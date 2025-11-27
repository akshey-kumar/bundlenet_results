import sys

sys.path.append(r'../')
import numpy as np
from functions import *

algorithm = 'tsne_time_delay_embedding'

'''
Best hyperparameters found were:
perplexity: 13.372368268498056
early_exaggeration: 31.68117393100013
n_iter: 2423.3311374373307
win: 10.861162643452484
'''
config ={
'perplexity': 13.372368268498056,
'early_exaggeration': 31.68117393100013,
'n_iter': 2423.3311374373307,
'win': 10.861162643452484,
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
        'RIBL', ]
    data = Database(data_set_no=worm_num)
    data.exclude_neurons(b_neurons)
    X = data.neuron_traces.T
    B = data.states
    state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing',
                   'Ventral turn']

    ### Preprocess and prepare data for BundLe Net
    time, X = preprocess_data(X, float(data.fps))
    X_, B_ = prep_data(X, B, win=int(config["win"]))

    ## Train test split
    X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)
    X0_tr = X_train[:, 0, :, :].reshape(X_train.shape[0], -1)
    X1_tr = X_train[:, 1, :, :].reshape(X_train.shape[0], -1)
    X0_tst = X_test[:, 0, :, :].reshape(X_test.shape[0], -1)
    X1_tst = X_test[:, 1, :, :].reshape(X_test.shape[0], -1)

    ### Deploy tsne
    dim = 3
    tsne = TSNE(
        n_components=dim,
        init = 'pca',
        perplexity = config["perplexity"],
        early_exaggeration = config["early_exaggeration"],
        n_iter = round(config["n_iter"]),
    )

    ### Projecting into latent space
    Y0_tr = tsne.fit_transform(X0_tr)
    Y1_tr = tsne.fit_transform(X1_tr)

    '''tsne does not have transform method alone only fittransform. 
    Hence it cannot be run separately on held out test data. So we
    set the test data equal to the train data'''
    Y0_tst = Y0_tr.copy()
    Y1_tst = Y1_tr.copy()
    B_test_1 = B_train_1.copy()

    # Save the weights
    # model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
    np.savetxt('data/generated/saved_Y/Y0_tr__' + algorithm + '_worm_' + str(worm_num), Y0_tr)
    np.savetxt('data/generated/saved_Y/Y1_tr__' + algorithm + '_worm_' + str(worm_num), Y1_tr)
    np.savetxt('data/generated/saved_Y/Y0_tst__' + algorithm + '_worm_' + str(worm_num), Y0_tst)
    np.savetxt('data/generated/saved_Y/Y1_tst__' + algorithm + '_worm_' + str(worm_num), Y1_tst)
    np.savetxt('data/generated/saved_Y/B_train_1__' + algorithm + '_worm_' + str(worm_num), B_train_1)
    np.savetxt('data/generated/saved_Y/B_test_1__' + algorithm + '_worm_' + str(worm_num), B_test_1)
