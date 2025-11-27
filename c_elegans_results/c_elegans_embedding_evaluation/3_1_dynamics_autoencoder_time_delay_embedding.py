import sys

sys.path.append(r'../')
import numpy as np
from functions import *
from tensorflow.keras import layers, Model
import tensorflow as tf

algorithm = 'dynamics_autoencoder_time_delay_embedding'
'''
Best hyperparameters found were:
lr: 0.0014057882878623295
epochs: 109.2968251007042
batch_size: 96.0
win: 12.388713412623767
layers_idx: 3.78631568228178
'''
config = {
    'lr': 0.0014057882878623295,
    'epochs': 109.2968251007042,
    'batch_size': 96.0,
    'win': 12.388713412623767,
    'layers_idx': 3.78631568228178,
}
# dynamics autoencoder architectures (encoder)
architectures = [
    [50, 10],  # Shallow Architecture
    [100, 150, 50, 10],  # Increasing then Decreasing Architecture
    [50, 30, 25, 10],  # Same encoder as in BunDLeNet
    [100, 80, 60, 40, 20, 10, 10],  # Deep Architecture
    [75, 25, 10]
]
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

    ### Train test split
    X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)
    X0_tr = X_train[:, 0, :, :]
    X1_tr = X_train[:, 1, :, :]
    Xdiff_tr = X1_tr - X0_tr
    X0_tst = X_test[:, 0, :, :]
    X1_tst = X_test[:, 1, :, :]
    Xdiff_tst = X1_tst - X0_tst

    ### Scaling input and output data
    Xdmax = (np.abs(Xdiff_tr)).max()  # Parameters for scaling
    Xdiff_tr, Xdiff_tst = Xdiff_tr / Xdmax, Xdiff_tst / Xdmax


    ### dynamics autoencoder architecture
    layers_idx = int(config["layers_idx"])
    encoder_layers = architectures[layers_idx]
    class Autoencoder(Model):
        def __init__(self, latent_dim=3):
            super(Autoencoder, self).__init__()
            self.latent_dim = latent_dim

            encoder_layers_list = [layers.Flatten()]
            for units in encoder_layers:
                encoder_layers_list.append(layers.Dense(units, activation='relu'))
            encoder_layers_list.append(layers.Dense(latent_dim, activation='relu'))  # latent layer

            self.encoder = tf.keras.Sequential(encoder_layers_list)

            decoder_layers_list = []
            # decoder symmetric to encoder except last layer reshaping
            for units in reversed(encoder_layers):
                decoder_layers_list.append(layers.Dense(units, activation='relu'))
            decoder_layers_list.append(layers.Dense(X0_tr.shape[-1] * X0_tr.shape[-2], activation='linear'))
            decoder_layers_list.append(layers.Reshape(X0_tr.shape[1:]))

            self.decoder = tf.keras.Sequential(decoder_layers_list)

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


    dynamics_autoencoder = Autoencoder(latent_dim=3)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=config["lr"])
    dynamics_autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse'])

    history = dynamics_autoencoder.fit(
        X0_tr,
        Xdiff_tr,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        validation_data=(X0_tst, Xdiff_tst),
        verbose=False
    )

    ### Predictions
    Xdiff_tr_pred = dynamics_autoencoder(X0_tr).numpy()
    Xdiff_tst_pred = dynamics_autoencoder(X0_tst).numpy()

    # Inverse scaling the data
    Xdiff_tr_pred, Xdiff_tr = Xdiff_tr_pred * Xdmax, Xdiff_tr * Xdmax
    Xdiff_tst_pred, Xdiff_tst = Xdiff_tst_pred * Xdmax, Xdiff_tst * Xdmax

    X1_tr_pred = X0_tr + Xdiff_tr_pred
    X1_tst_pred = X0_tst + Xdiff_tst_pred

    baseline_tst = mean_squared_error(flat_partial(X1_tst), flat_partial(X0_tst))
    modelmse_tst = mean_squared_error(flat_partial(X1_tst), flat_partial(X1_tst_pred))

    ### Projecting into latent space
    Y0_tr = dynamics_autoencoder.encoder(X0_tr).numpy()
    Y1_tr = dynamics_autoencoder.encoder(X1_tr).numpy()

    Y0_tst = dynamics_autoencoder.encoder(X0_tst).numpy()
    Y1_tst = dynamics_autoencoder.encoder(X1_tst).numpy()

    # Save the weights
    # model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
    np.savetxt('data/generated/saved_Y/Y0_tr__' + algorithm + '_worm_' + str(worm_num), Y0_tr)
    np.savetxt('data/generated/saved_Y/Y1_tr__' + algorithm + '_worm_' + str(worm_num), Y1_tr)
    np.savetxt('data/generated/saved_Y/Y0_tst__' + algorithm + '_worm_' + str(worm_num), Y0_tst)
    np.savetxt('data/generated/saved_Y/Y1_tst__' + algorithm + '_worm_' + str(worm_num), Y1_tst)
    np.savetxt('data/generated/saved_Y/B_train_1__' + algorithm + '_worm_' + str(worm_num), B_train_1)
    np.savetxt('data/generated/saved_Y/B_test_1__' + algorithm + '_worm_' + str(worm_num), B_test_1)





