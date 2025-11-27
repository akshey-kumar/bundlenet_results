import os
import numpy as np
from sklearn.metrics import mean_squared_error
from c_elegans_embedding_evaluation.functions import Database, preprocess_data, prep_data, timeseries_train_test_split, plot_latent_timeseries, plot_phase_space

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session

algorithm='rnn_autoencoder'

# load data
worm_num = 0
b_neurons = ['AVAR', 'AVAL', 'SMDVR', 'SMDVL', 'SMDDR', 'SMDDL', 'RIBR', 'RIBL']
data = Database(data_set_no=worm_num)
data.exclude_neurons(b_neurons)
x = data.neuron_traces.T
b = data.states

time, x = preprocess_data(x, float(data.fps))

# hyperparameter tuning function
def train_rnn_autoencoder(config):
    # prepare data with given  window size
    x_, b_ = prep_data(x, b, win=int(config["win"]))

    # train test split
    x_train, x_test, b_train, b_test = timeseries_train_test_split(x_, b_)
    x0_tr = x_train[:, 0, :, :]
    x0_tst = x_test[:, 0, :, :]
    x1_tr = x_train[:, 1, :, :]
    x1_tst = x_test[:, 1, :, :]

    # RNN autoencoder model
    class RnnAutoencoder(tf.keras.Model):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super(RnnAutoencoder, self).__init__()
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim
            # Encoder GRU
            self.encoder_rnn = layers.GRU(hidden_dim, return_sequences=True, return_state=True)
            # projection to/from latent
            self.to_embedding = layers.Dense(latent_dim)
            self.from_embedding = layers.Dense(hidden_dim)
            # Generator (decoder)
            self.decoder_rnn = layers.GRU(hidden_dim, return_sequences=True, return_state=True)
            # Output projection
            self.readout = layers.Dense(input_dim)

        def encode(self, x):
            _, h_enc = self.encoder_rnn(x)
            embedding = self.to_embedding(h_enc)
            return embedding

        def decode(self, embedding, batch_size, seq_len):
            # Project latent embedding to initial hidden state of decoder
            h0 = self.from_embedding(embedding)
            decoder_input = tf.zeros((batch_size, seq_len, self.hidden_dim))
            decoded_seq, _ = self.decoder_rnn(decoder_input, initial_state=h0)
            decoder_output = self.readout(decoded_seq)
            return decoder_output

        def call(self, x):
            batch_size = tf.shape(x)[0]
            seq_len = tf.shape(x)[1]
            embedding = self.encode(x)
            reconstructed = self.decode(embedding, batch_size, seq_len)
            return reconstructed

    # model initialization
    # Instantiate model
    model = RnnAutoencoder(
        input_dim=x0_tr.shape[-1],
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=3,
    )

    model.compile(optimizer=optimizers.Adam(learning_rate=config["lr"]), loss='mse')
    model.fit(x0_tr, x0_tr, batch_size=int(config["batch_size"]), epochs=int(config["epochs"]), verbose=True)

    # evaluation on test data
    reconstructed = model.predict(x0_tst)
    model_mse = mean_squared_error(x0_tst.reshape(len(x0_tst), -1), reconstructed.reshape(len(reconstructed), -1))
    print('Reconstructed shape:', reconstructed.shape)
    print('MSE:', round(model_mse, 8))

    # report result to ray tune
    session.report({"mse": model_mse})

if __name__ == "__main__":

    max_epochs = 2000
    # Hyperparameter search space
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "epochs": tune.loguniform(10, max_epochs, 1),
        "batch_size": tune.qloguniform(32, 512, 16),
        "win": tune.loguniform(1, 25),
        "hidden_dim": tune.qloguniform(16, 128, 16)
    }

    # hyperparameter tuning
    search_algo = HyperOptSearch(metric="mse", mode="min")

    tuner = tune.Tuner(
        tune.with_parameters(train_rnn_autoencoder),
        tune_config=tune.TuneConfig(
            search_alg=search_algo,
            num_samples=200,
            max_concurrent_trials=7
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric='mse', mode='min')
    print("Minimum validation loss:", best_result.metrics['mse'])
    print("Best hyperparameters found were: ", best_result.config)

    # save best hyperparameters to a file
    results_dir = 'c_elegans_embedding_evaluation/hyperparameter_optimisation/optimal_hyperparameters/'
    os.makedirs(results_dir, exist_ok=True)

    best_params_path = os.path.join(results_dir, f"best_params_c_elegans_{worm_num}_{algorithm}.txt")
    with open(best_params_path, 'w') as f:
        f.write(f"Min loss: {best_result.metrics['mse']}\n")
        f.write("Best hyperparameters found were:\n")
        for param, value in best_result.config.items():
            f.write(f"{param}: {value}\n")
