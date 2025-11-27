import os
from ray import tune
#from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch

from c_elegans_embedding_evaluation.functions import Database, preprocess_data, prep_data, timeseries_train_test_split
from ray.air import session
from sklearn.manifold import TSNE


### Load Data (and excluding behavioural neurons)
worm_num = 0
algorithm = 'tsne'
b_neurons = ['AVAR', 'AVAL', 'SMDVR', 'SMDVL', 'SMDDR', 'SMDDL', 'RIBR', 'RIBL']
data = Database(data_set_no=worm_num)
data.exclude_neurons(b_neurons)
x = data.neuron_traces.T
b = data.states
state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']

time, x = preprocess_data(x, float(data.fps))

# hyperparameter tuning function
def train_tsne(config):
    # prepare data with given window size
    x_, b_ = prep_data(x, b, win=round(config["win"]))
    x0_ = x_[:, 0, :, :].reshape(x_.shape[0], -1)

    tsne = TSNE(
        n_components=3,
        init='pca',
        perplexity=config["perplexity"],
        early_exaggeration=config["early_exaggeration"],
        n_iter=round(config["n_iter"])
    )
    y0_ = tsne.fit_transform(x0_)
    session.report({"loss":tsne.kl_divergence_})


if __name__ == "__main__":
    # Hyperparameter search space
    search_space = {
        'perplexity': tune.uniform(5,50),
        'early_exaggeration': tune.uniform(12,32),
        'n_iter': tune.uniform(250,2500),
        'win': tune.loguniform(1, 20)
    }

    # hyperparameter tuning
    search_algo = HyperOptSearch(metric="loss", mode="min")
    tuner = tune.Tuner(
        tune.with_parameters(train_tsne),
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
    print("Best hyperparameters found were: ", best_result.config)


    # save best hyperparameters to a file
    results_dir = 'c_elegans_embedding_evaluation/hyperparameter_optimisation/optimal_hyperparameters/'
    os.makedirs(results_dir, exist_ok=True)

    best_params_path = os.path.join(results_dir, f"best_params_c_elegans_{worm_num}_{algorithm}.txt")
    with open(best_params_path, 'w') as f:
        f.write(f"Min loss: {best_result.metrics['loss']}\n")
        f.write("Best hyperparameters found were:\n")
        for param, value in best_result.config.items():
            f.write(f"{param}: {value}\n")