import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

# Load Data (excluding behavioural neurons) and plot
worm_num = 0
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

for gamma in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.99, 0.999]:
    print(f'gamma: {gamma}')
    algorithm = 'bundlenet_ablation_study'
    if gamma==0.0:
        algorithm += '_purely_behavioural'
    elif gamma==1.0:
        algorithm += '_purely_dynamical'

    # deploy BunDLe Net
    model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names), input_shape=x_.shape)
    loss_array, _ = train_model(
        x_,
        b_,
        model,
        b_type='discrete',
        gamma=gamma,
        learning_rate=0.001,
        n_epochs=1000,
        initialisation='best_of_5_init',
    )
    plt.figure()
    for i, label in enumerate([
        r"$\mathcal{L}_{\mathrm{Markov}}$",
        r"$\mathcal{L}_{\mathrm{Behavior}}$",
        r"Total loss $\mathcal{L}$"
    ]):
        plt.plot(loss_array[:, i], label=label)
    plt.legend()

    # projecting into latent space
    y0_ = project_into_latent_space(x_[:,0], model)

    # save the embeddings
    save_model = False
    if save_model:
        np.savetxt(f'data/generated/ablation_studies/y0__{algorithm}_worm_{worm_num}_gamma_{gamma}', y0_)
        np.savetxt(f'data/generated/ablation_studies/b__{algorithm}_worm_{worm_num}_gamma_{gamma}', b_)
        y0_ = np.loadtxt(f'data/generated/ablation_studies/y0__{algorithm}_worm_{worm_num}_gamma_{gamma}')
        b_ = np.loadtxt(f'data/generated/ablation_studies/b__{algorithm}_worm_{worm_num}_gamma_{gamma}').astype(int)

    # Plotting latent space dynamics
    vis = LatentSpaceVisualiser(y0_, b_, data.behaviour_names, show_points=True)
    # vis.plot_latent_timeseries()
    vis.plot_phase_space()
    # vis.rotating_plot(
    #     filename=f'c_elegans_results/ablation_studies/figures/rotation_{algorithm}_worm_{worm_num}_gamma_{gamma}.gif',
    #     show_fig=False,
    #     #arrow_length_ratio=0.01
    #     )

plt.show()