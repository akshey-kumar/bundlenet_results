import numpy as np
import matplotlib.pyplot as plt

from functions import BunDLeNet, train_model, preprocess_data, prep_data, Database, timeseries_train_test_split
import tensorflow as tf

# load Data (excluding behavioural neurons) and plot
for worm_num in range(5):
    algorithm = 'bundlenet_tf'
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
    data = Database(data_path=data_path, data_set_no=worm_num)
    data.exclude_neurons(b_neurons)
    x = data.neuron_traces.T
    b = data.states

    # prepare data for BunDLe Net
    _, x = preprocess_data(x, float(data.fps))
    x_, b_ = prep_data(x, b, win=15)
    x_train, x_test, b_train, b_test = timeseries_train_test_split(x_, b_)

    # deploy BunDLe Net
    model = BunDLeNet(latent_dim=3)
    model.build(input_shape=x_train.shape)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    loss_array = train_model(x_train,
                             b_train,
                             model,
                             optimizer,
                             gamma=0.9,
                             n_epochs=2000,
                             pca_init=False,
                             best_of_5_init=True
                             )


    ### Projecting into latent space
    y0_tr = model.tau(x_train[:, 0]).numpy()
    y1_tr = model.tau(x_train[:, 1]).numpy()
    y0_tst = model.tau(x_test[:, 0]).numpy()
    y1_tst = model.tau(x_test[:, 1]).numpy()

    # saving
    save_model = True
    if save_model:
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y0_tr__{algorithm}_worm_{worm_num}', y0_tr)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y1_tr__{algorithm}_worm_{worm_num}', y1_tr)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y0_tst__{algorithm}_worm_{worm_num}', y0_tst)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y1_tst__{algorithm}_worm_{worm_num}', y1_tst)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/b_tr__{algorithm}_worm_{worm_num}', b_train)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/b_tst__{algorithm}_worm_{worm_num}', b_test)

    # plotting latent space dynamics
    #vis = LatentSpaceVisualiser(y0_, b_, data.behaviour_names)
    #vis.plot_latent_timeseries()
    #vis.plot_phase_space()




