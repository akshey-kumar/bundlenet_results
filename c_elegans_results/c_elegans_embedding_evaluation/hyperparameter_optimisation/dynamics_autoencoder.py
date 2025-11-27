import os
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from c_elegans_embedding_evaluation.functions import Database, preprocess_data, prep_data, timeseries_train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import mean_squared_error
import numpy as np
from ray.air import session


algorithm = 'dynamics_autoencoder'

worm_num = 0
b_neurons = ['AVAR', 'AVAL', 'SMDVR', 'SMDVL', 'SMDDR', 'SMDDL', 'RIBR', 'RIBL']
data = Database(data_set_no=worm_num)
data.exclude_neurons(b_neurons)
x = data.neuron_traces.T
b = data.states

time, x = preprocess_data(x, float(data.fps))

# dynamics autoencoder architectures (encoder)
architectures = [
    [50, 10],  # Shallow Architecture
    [100, 150, 50, 10],  # Increasing then Decreasing Architecture
    [50, 30, 25, 10],  # Same encoder as in BunDLeNet
    [100, 80, 60, 40, 20, 10, 10],  # Deep Architecture
    [75, 25, 10]
]

def flat_partial(x):
    return x.reshape(x.shape[0], -1)

# hyperparameter tuning function
def train_dynamics_autoencoder(config):

    # Prepare data
    x_, b_ = prep_data(x, b, win=round(config["win"]))
    x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)
    x0_tr = x_train[:, 0, :, :]
    x1_tr = x_train[:, 1, :, :]
    xdiff_tr = x1_tr - x0_tr
    x0_tst = x_test[:, 0, :, :]
    x1_tst = x_test[:, 1, :, :]
    xdiff_tst = x1_tst - x0_tst

    # Scaling input and output data
    xdmax = (np.abs(xdiff_tr)).max()
    xdiff_tr, xdiff_tst = xdiff_tr / xdmax, xdiff_tst / xdmax

    # define autoencoder - flexible based on selected architecture
    layers_idx = int(config["layers_idx"])
    encoder_layers = architectures[layers_idx]

    # Define Autoencoder dynamically based on selected architecture
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
            decoder_layers_list.append(layers.Dense(x0_tr.shape[-1] * x0_tr.shape[-2], activation='linear'))
            decoder_layers_list.append(layers.Reshape(x0_tr.shape[1:]))

            self.decoder = tf.keras.Sequential(decoder_layers_list)

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    dynamics_autoencoder = Autoencoder(latent_dim=3)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=config["lr"])
    dynamics_autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse'])

    history = dynamics_autoencoder.fit(
        x0_tr,
        xdiff_tr,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        validation_data=(x0_tst, xdiff_tst),
        verbose=False
    )

    xdiff_tst_pred = dynamics_autoencoder(x0_tst).numpy()
    xdiff_tst_pred = xdiff_tst_pred * xdmax
    x1_tst_pred = x0_tst + xdiff_tst_pred

    model_mse_tst = mean_squared_error(flat_partial(x1_tst), flat_partial(x1_tst_pred))
    session.report({"loss": model_mse_tst})

if __name__ == "__main__":

    max_epochs = 1000
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "epochs": tune.loguniform(10, max_epochs),
        "batch_size": tune.qloguniform(32, 512, 16),
        "win": tune.loguniform(1, 20),
        "layers_idx": tune.uniform(0, 4),
    }

    # hyperparameter tuning
    search_algo = HyperOptSearch(metric="loss", mode="min")
    tuner = tune.Tuner(
        tune.with_parameters(train_dynamics_autoencoder),
        tune_config=tune.TuneConfig(
            search_alg=search_algo,
            num_samples=200,
            max_concurrent_trials=7
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric='loss', mode='min')
    print("Minimum validation loss:", best_result.metrics['loss'])
    print("best hyperparameters found were:", best_result.config)

    # save best hyperparameters to a file
    results_dir = 'c_elegans_embedding_evaluation/hyperparameter_optimisation/optimal_hyperparameters/'
    os.makedirs(results_dir, exist_ok=True)

    best_params_path = os.path.join(results_dir, f"best_params_c_elegans_{worm_num}_{algorithm}.txt")
    with open(best_params_path, 'w') as f:
        f.write(f"Min loss: {best_result.metrics['loss']}\n")
        f.write("Best hyperparameters found were:\n")
        for param, value in best_result.config.items():
            f.write(f"{param}: {value}\n")