import os
import numpy as np
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data

# load data
worm_num = 0
algorithm = 'bundlenet'
b_neurons = ['AVAR', 'AVAL', 'SMDVR', 'SMDVL', 'SMDDR', 'SMDDL', 'RIBR', 'RIBL']
data_path = 'data/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
data.exclude_neurons(b_neurons)
x = data.neuron_traces.T
b = data.behaviour

def train_bundlenet(config):
    # Prepare data with given window size
    x_, b_ = prep_data(x, b, win=int(config['win']))
    model = BunDLeNet(
        latent_dim=3,
        num_behaviour=len(data.behaviour_names),
        input_shape=x_.shape
    )
    loss_array, _ = train_model(
        x_,
        b_,
        model,
        b_type='discrete',
        gamma=config['gamma'],
        learning_rate=config['learning_rate'],
        n_epochs=int(config['epochs'])
    )
    # Use the last total loss as metric (assumes total loss is last column)
    min_loss = float(np.min(loss_array[:, -1]))
    tune.report(min_loss=min_loss)

if __name__ == "__main__":
    max_epochs = 1000
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "epochs": tune.loguniform(10, max_epochs),
    }
    search_algo = HyperOptSearch(metric="min_loss", mode="min")
    tuner = tune.Tuner(
        tune.with_parameters(train_bundlenet),
        tune_config=tune.TuneConfig(
            search_alg=search_algo,
            num_samples=50,
            max_concurrent_trials=4
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric='min_loss', mode='min')
    print("Minimum total loss:", best_result.metrics['min_loss'])
    print("Best hyperparameters found were: ", best_result.config)

    # Save best hyperparameters to a file
    results_dir = 'data/generated/optimal_hyperparameters/'
    os.makedirs(results_dir, exist_ok=True)
    best_params_path = os.path.join(results_dir, f"best_params_c_elegans_{worm_num}_{algorithm}.txt")
    with open(best_params_path, 'w') as f:
        f.write(f"Minimum min_loss: {best_result.metrics['min_loss']}\n")
        f.write("Best hyperparameters found were:\n")
        for param, value in best_result.config.items():
            f.write(f"{param}: {value}\n")
            
            
