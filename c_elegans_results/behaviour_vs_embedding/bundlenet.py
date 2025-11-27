import os

import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

# load Data (excluding behavioural neurons) and plot
for worm_num in range(5):
    algorithm = 'bundlenet'
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

    # prepare data for BunDLe Net
    x_, b_ = prep_data(x, b, win=1)
    x_train, x_test, b_train, b_test = timeseries_train_test_split(x_, b_)

    # Deploy BunDLe Net
    model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names), input_shape=x_.shape)
    loss_array, _ = train_model(
        x_train,
        b_train,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=1500,
        validation_data=(x_test, b_test),
        initialisation='best_of_5_init',
    )

    # projecting into latent space
    y0_train = project_into_latent_space(x_train[:,0], model)
    y0_test = project_into_latent_space(x_test[:,0], model)
    y0_ = project_into_latent_space(x_[:,0], model)

    # save the weights
    save_model = True
    if save_model:
        os.makedirs('data/generated/behaviour_vs_embedding/c_elegans/', exist_ok=True)
        np.savetxt(f'data/generated/behaviour_vs_embedding/c_elegans/y0_train_{algorithm}_worm_{worm_num}', y0_train)
        np.savetxt(f'data/generated/behaviour_vs_embedding/c_elegans/y0_test_{algorithm}_worm_{worm_num}', y0_test)
        np.savetxt(f'data/generated/behaviour_vs_embedding/c_elegans/y0__{algorithm}_worm_{worm_num}', y0_)

        np.savetxt(f'data/generated/behaviour_vs_embedding/c_elegans/b_train_{algorithm}_worm_{worm_num}', b_train)
        np.savetxt(f'data/generated/behaviour_vs_embedding/c_elegans/b_test_{algorithm}_worm_{worm_num}', b_test)
        np.savetxt(f'data/generated/behaviour_vs_embedding/c_elegans/b__{algorithm}_worm_{worm_num}', b_)

    # plotting latent space dynamics
    vis = LatentSpaceVisualiser(y0_, b_, data.behaviour_names)
    vis.plot_latent_timeseries()
    #vis.plot_phase_space()
