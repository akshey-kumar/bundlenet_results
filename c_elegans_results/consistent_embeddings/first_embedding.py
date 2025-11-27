import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
import torch

# load Data (excluding behavioural neurons) and plot
worm_num = 0
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
x_, b_ = prep_data(x, b, win=15)

# Deploy BunDLe Net
model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names), input_shape=x_.shape)
loss_array, _ = train_model(
    x_,
    b_,
    model,
    b_type='discrete',
    gamma=0.9,
    learning_rate=0.001,
    n_epochs=2000,
    initialisation='best_of_5_init',
)

for i, label in enumerate([
    r"$\mathcal{L}_{\mathrm{Markov}}$",
    r"$\mathcal{L}_{\mathrm{Behavior}}$",
    r"Total loss $\mathcal{L}$"
]):
    plt.plot(loss_array[:, i], label=label)
plt.legend()
plt.show()

# projecting into latent space
y0_ = project_into_latent_space(x_[:,0], model)

# save the entire model
save_model = True
model_path = f'data/generated/pytorch_models/c_elegans/bundle_net_model_worm_{worm_num}.pt'
if save_model:
    torch.save(model, model_path)

# plotting latent space dynamics
#vis = LatentSpaceVisualiser(y0_, b_, data.behaviour_names)
#vis.plot_latent_timeseries()
#vis.plot_phase_space()
