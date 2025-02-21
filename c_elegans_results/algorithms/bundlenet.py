import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, model_inference
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

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
x_, b_ = prep_data(x, b, win=1)

# Deploy BunDLe Net
model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names), input_shape=x_.shape)
loss_array, _ = train_model(
    x_,
    b_,
    model,
    b_type='discrete',
    gamma=0.9,
    learning_rate=0.001,
    n_epochs=1000
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
y0_ = model_inference(x_, model)

# save the weights
save_model = False
if save_model:
    #model.save_weights(f'data/generated/models/BunDLeNet_model_worm_{worm_num}')
    np.savetxt(f'data/generated/embeddings/y0__{algorithm}_worm_{worm_num}', y0_)
    np.savetxt(f'data/generated/embeddings/b__{algorithm}_worm_{worm_num}', b_)
    y0_ = np.loadtxt(f'data/generated/embeddings/y0__{algorithm}_worm_{worm_num}')
    b_ = np.loadtxt(f'data/generated/embeddings/b__{algorithm}_worm_{worm_num}').astype(int)

# plotting latent space dynamics
vis = LatentSpaceVisualiser(y0_, b_, data.behaviour_names)
vis.plot_latent_timeseries()
vis.plot_phase_space()
#vis.rotating_plot(filename='figures/rotation_' + algorithm + '_worm_' + str(worm_num) + '.gif')
