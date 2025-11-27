import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split

algorithm = 'BunDLeNet_temporal_shuffling'
os.makedirs('data/generated/shuffling_experiments', exist_ok=True)
for worm_num in range(5):
    # Load Data (and excluding behavioural neurons)
    b_neurons = [
        'AVAR',
        'AVAL',
        'SMDVR',
        'SMDVL',
        'SMDDR',
        'SMDDL',
        'RIBR',
        'RIBL', ]
    data_path = 'data/raw/c_elegans/NoStim_Data.mat'
    data = Database(data_path=data_path, dataset_no=worm_num)
    data.exclude_neurons(b_neurons)
    X = data.neuron_traces.T
    B = data.behaviour

    # Shuffling data to destroy dynamical information while keeping behavioural information
    rand_idx = np.random.permutation(X.shape[0])
    X, B = X[rand_idx], B[rand_idx]

    # prepare data for BunDLe Net
    X_, B_ = prep_data(X, B, win=1)
    X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)


    model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names), input_shape=X_train.shape)
    train_history, test_history = train_model(
        X_train,
        B_train_1,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=1000,
        validation_data=(X_test, B_test_1),
        initialisation='best_of_5_init',
    )
    history = {
        "markov_train_loss": train_history[:, 0],
        "markov_test_loss": test_history[:, 0],
        "behaviour_train_loss": train_history[:, 1],
        "behaviour_test_loss": test_history[:, 1],
        "total_train_loss": train_history[:, -1],
        "total_test_loss": test_history[:, -1]
    }
    np.save(f'data/generated/shuffling_experiments/learning_curves_{algorithm}_{worm_num}.npy', history)

    ## Projecting into latent space
    Y0_ = project_into_latent_space(X_[:, 0], model)

    # Save the weights

    np.savetxt(f'data/generated/shuffling_experiments/Y0__{algorithm}_worm_{worm_num}', Y0_)
    np.savetxt(f'data/generated/shuffling_experiments/B__{algorithm}_worm_{worm_num}', B_)

