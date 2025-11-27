import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural

import tensorflow as tf

def fit_inverse_embedder(x0_tr, y0_tr, x0_tst, y0_tst, plot_history=False):
    print(x0_tr.max(), y0_tr.max(), x0_tr.min(), y0_tr.min())
    # interpolating as much as possible
    inverse_embedder = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(x0_tr.shape[1], activation='linear')
    ])
    inverse_embedder.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mse', tf.keras.metrics.R2Score()]
    )
    print(y0_tr.shape, x0_tr.shape)
    history = inverse_embedder.fit(
        y0_tr,
        x0_tr,
        epochs=1000,
        batch_size=100,
        validation_data=(y0_tst, x0_tst),
        verbose=0
    )

    if plot_history:
        # Plot training & validation loss values
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')

        # Adding titles and labels
        plt.title('Model Loss Over Epochs')
        plt.ylabel('Loss (MSE)')
        plt.xlabel('Epochs')
        plt.legend(loc='upper right')
        plt.grid(True)

        # Plot metrics
        print(history.history.keys())
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['r2_score'], label='Training R2', color='blue')
        plt.plot(history.history['val_r2_score'], label='Validation R2', color='orange')

        # Adding titles and labels
        plt.title('R2 Over Epochs')
        plt.ylabel('R2)')
        plt.xlabel('Epochs')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

    return inverse_embedder

if __name__ == '__main__':
    algorithm = 'BunDLeNet'
    worm_num = 0

    data_path = 'data/raw/c_elegans/NoStim_Data.mat'
    data = Database(data_path=data_path, dataset_no=worm_num)
    b_neurons = [
        'AVAR',
        'AVAL',
        'SMDVR',
        'SMDVL',
        'SMDDR',
        'SMDDL',
        'RIBR',
        'RIBL', ]
    data.exclude_neurons(b_neurons)
    x = data.neuron_traces.T
    b = data.behaviour
    x_, b_ = prep_data(x, b, win=15)
    x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)
    x1_tr = x_train[:, 1, -1, :]
    x1_tst = x_test[:, 1, -1, :]

    file_pattern = f'data/generated/saved_Y/{{}}__{algorithm}_worm_{worm_num}'
    y1_tr = np.loadtxt(file_pattern.format('Y1_tr'))
    y1_tst = np.loadtxt(file_pattern.format('Y1_tst'))
    b_train_1 = np.loadtxt(file_pattern.format('B_train_1')).astype(int)
    b_test_1 = np.loadtxt(file_pattern.format('B_test_1')).astype(int)

    print(x1_tr.max(), y1_tr.max(), x1_tr.min(), y1_tr.min())
    inverse_embedder = fit_inverse_embedder(x1_tr, y1_tr, x1_tst, y1_tst)

    x1_tr_pred = inverse_embedder(y1_tr).numpy()
    x1_tst_pred = inverse_embedder(y1_tst).numpy()
    # plotting_neuronal_behavioural(x1_tr, b_train_1, b_names=data.behaviour_names)
    # plotting_neuronal_behavioural(x1_tr_pred, b_train_1, b_names=data.behaviour_names)


    plt.figure(figsize=(6, 5))
    for i in range(35, 45):
        plt.plot(x1_tr[500:2000, i] + i / 2)
    plt.figure(figsize=(6, 5))
    for i in range(35, 45):
        plt.plot(x1_tr_pred[500:2000, i] + i / 2)

    plt.show()


    plt.figure(figsize=(6, 5))
    for i in range(35, 45):
        plt.plot(x1_tst[:, i] + i / 2)
    plt.figure(figsize=(6, 5))
    for i in range(35, 45):
        plt.plot(x1_tst_pred[:, i] + i / 2)

    plt.show()