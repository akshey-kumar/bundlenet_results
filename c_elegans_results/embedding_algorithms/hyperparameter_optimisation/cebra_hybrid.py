import os
import numpy as np
from sklearn.metrics import mean_squared_error
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from cebra import CEBRA
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
from ray.tune.search.hyperopt import HyperOptSearch

# load data
worm_num = 0
algorithm = 'cebra_hybrid'
b_neurons = ['AVAR', 'AVAL', 'SMDVR', 'SMDVL', 'SMDDR', 'SMDDL', 'RIBR', 'RIBL']
data_path = 'data/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
data.exclude_neurons(b_neurons)
x = data.neuron_traces.T
b = data.behaviour

# hyperparameter tuning function
def train_cebra_hybrid(config):
    # prepare data with given window size
    x_, b_ = prep_data(x, b, win=1)
    x_train, x_test, b_train, b_test = timeseries_train_test_split(x_, b_)
    x_train = x_train[:, 1, :, :].reshape(x_train.shape[0], -1)
    x_test = x_test[:, 1, :, :].reshape(x_test.shape[0], -1)

    # fit CEBRA hybrid
    cebra_hybrid_model = CEBRA(
        model_architecture=config["model_architecture"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        temperature=config["temperature"],
        output_dimension=3,
        max_iterations=config["max_iterations"],
        distance=config["distance"],
        conditional='time_delta',
        device='cuda_if_available',
        verbose=True,
        time_offsets=config["time_offsets"],
        hybrid=True
    )
    cebra_hybrid_model.fit(x_train, b_train.astype(float))

    # projecting into latent space
    y_train = cebra_hybrid_model.transform(x_train)
    y_test = cebra_hybrid_model.transform(x_test)

    # evaluation
    def behaviour_decoding_accuracy(y_train, y_test, b_train, b_test, n_neighbors=36):
        behaviour_decoder = KNeighborsClassifier(n_neighbors, metric='cosine')
        behaviour_decoder.fit(y_train, b_train)
        b_pred = behaviour_decoder.predict(y_test)

        test_accuracy = sklearn.metrics.accuracy_score(b_test, b_pred)
        return test_accuracy

    test_acc = behaviour_decoding_accuracy(y_train, y_test, b_train, b_test)

    # report result to ray tune
    tune.report({"test_acc":test_acc})

if __name__ == "__main__":

    max_epochs = 500
    # Hyperparameter search space
    search_space = {
        "model_architecture": tune.choice(['offset10-model', 'offset1-model-mse']),
        "batch_size": tune.choice([256, 512]),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "temperature": tune.uniform(0.1, 1.0),
        "max_iterations": tune.choice([2000, 5000, 10000]),
        "distance": tune.choice(['cosine', 'euclidean']),
        "time_offsets": tune.choice([5, 10, 15, 20]),
    }

    # hyperparameter tuning
    search_algo = HyperOptSearch(metric="test_acc", mode="min")

    tuner = tune.Tuner(
        tune.with_parameters(train_cebra_hybrid),
        tune_config=tune.TuneConfig(
            search_alg=search_algo,
            num_samples=200,
            max_concurrent_trials=7
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric='test_acc', mode='min')
    print("Minimum validation loss:", best_result.metrics['test_acc'])
    print("Best hyperparameters found were: ", best_result.config)

    # save best hyperparameters to a file
    results_dir = 'data/generated/optimal_hyperparameters/'
    os.makedirs(results_dir, exist_ok=True)

    best_params_path = os.path.join(results_dir, f"best_params_c_elegans_{worm_num}_{algorithm}.txt")
    with open(best_params_path, 'w') as f:
        f.write(f"Minimum test_acc: {best_result.metrics['test_acc']}\n")
        f.write("Best hyperparameters found were:\n")
        for param, value in best_result.config.items():
            f.write(f"{param}: {value}\n")

