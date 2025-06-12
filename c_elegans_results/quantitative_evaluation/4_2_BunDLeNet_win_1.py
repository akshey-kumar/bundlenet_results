import sys

sys.path.append(r'../')
import numpy as np
from functions import *
import matplotlib.pyplot as plt

algorithm = 'BunDLeNet_win_1'

# Load Data (and excluding behavioural neurons)
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
    time, X = preprocess_data(X, data.fps)
    X_, B_ = prep_data(X, B, win=1)
    print(X_.shape)

    ## Train test split
    X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)

    ### Deploy BunDLe Net
    model = BunDLeNet(latent_dim=3)
    model.build(input_shape=X_train.shape)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    loss_array = train_model(X_train,
                             B_train_1,
                             model,
                             optimizer,
                             gamma=0.9,
                             n_epochs=2000,
                             pca_init=False,
                             best_of_5_init=True
                             )

    ### Projecting into latent space
    Y0_tr = model.tau(X_train[:, 0]).numpy()
    Y1_tr = model.tau(X_train[:, 1]).numpy()

    Y0_tst = model.tau(X_test[:, 0]).numpy()
    Y1_tst = model.tau(X_test[:, 1]).numpy()

    # Save the weights
    # model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
    np.savetxt('data/generated/saved_Y/Y0_tr__' + algorithm + '_worm_' + str(worm_num), Y0_tr)
    np.savetxt('data/generated/saved_Y/Y1_tr__' + algorithm + '_worm_' + str(worm_num), Y1_tr)
    np.savetxt('data/generated/saved_Y/Y0_tst__' + algorithm + '_worm_' + str(worm_num), Y0_tst)
    np.savetxt('data/generated/saved_Y/Y1_tst__' + algorithm + '_worm_' + str(worm_num), Y1_tst)
    np.savetxt('data/generated/saved_Y/B_train_1__' + algorithm + '_worm_' + str(worm_num), B_train_1)
    np.savetxt('data/generated/saved_Y/B_test_1__' + algorithm + '_worm_' + str(worm_num), B_test_1)
