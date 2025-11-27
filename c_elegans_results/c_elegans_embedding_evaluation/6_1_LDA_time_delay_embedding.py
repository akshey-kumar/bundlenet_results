import sys
sys.path.append(r'../')
import numpy as np
from functions import *

algorithm = 'LDA_time_delay_embedding'

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
    X_, B_ = prep_data(X, B, win=15)

    ## Train test split
    X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)
    #X_train = X_train.reshape(X_train.shape[0], 2, 1, -1)
    #X_test = X_test.reshape(X_test.shape[0], 2, 1, -1)
    X0_tr = X_train[:, 0, :, :].reshape(X_train.shape[0], -1)
    X1_tr = X_train[:, 1, :, :].reshape(X_train.shape[0], -1)
    X0_tst = X_test[:, 0, :, :].reshape(X_test.shape[0], -1)
    X1_tst = X_test[:, 1, :, :].reshape(X_test.shape[0], -1)

    ### Deploy LDA
    dim = 3
    lda = LinearDiscriminantAnalysis(n_components=dim)
    lda.fit(X0_tr, B_train_1)
    print('Accuracy of LDA on train data', lda.score(X0_tr, B_train_1))
    print('Accuracy of LDA on test data', lda.score(X0_tst, B_test_1))

    ### Projecting into latent space
    Y0_tr = lda.transform(X0_tr)
    Y1_tr = lda.transform(X1_tr)
    Y0_tst = lda.transform(X0_tst)
    Y1_tst = lda.transform(X1_tst)

    # Save the weights
    # model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
    np.savetxt('data/generated/saved_Y/Y0_tr__' + algorithm + '_worm_' + str(worm_num), Y0_tr)
    np.savetxt('data/generated/saved_Y/Y1_tr__' + algorithm + '_worm_' + str(worm_num), Y1_tr)
    np.savetxt('data/generated/saved_Y/Y0_tst__' + algorithm + '_worm_' + str(worm_num), Y0_tst)
    np.savetxt('data/generated/saved_Y/Y1_tst__' + algorithm + '_worm_' + str(worm_num), Y1_tst)
    np.savetxt('data/generated/saved_Y/B_train_1__' + algorithm + '_worm_' + str(worm_num), B_train_1)
    np.savetxt('data/generated/saved_Y/B_test_1__' + algorithm + '_worm_' + str(worm_num), B_test_1)
