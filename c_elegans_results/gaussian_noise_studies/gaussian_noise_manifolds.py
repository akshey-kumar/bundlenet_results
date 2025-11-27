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

for gaussian_noise in [0.0, 0.001, 0.005, 0.01, 0.3, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
    print(f'gaussian_noise: {gaussian_noise}')
    algorithm = 'bundlenet_gaussian_noise_study'


    class BunDLeNet(nn.Module):
        """Behaviour and Dynamical Learning Network (BunDLeNet) model.

        This model represents the BunDLe Net's architecture for deep learning and is
        based on the commutativity diagrams. The resulting model is dynamically
        consistent (DC) and behaviourally consistent (BC) as per the notion described
        in the paper.

        Args:
            latent_dim (int):
                Dimension of the latent space.

            num_behaviour (int):
                For discrete-valued behaviours, this stipulates the number of
                discrete behavioural states
                For continuous-valued behaviours, this stipulates the number of
                behaviour variables

            input_shape (tuple):
                Shape of the input data.

        """

        def __init__(self, latent_dim: int, num_behaviour: int, input_shape: tuple):
            super(BunDLeNet, self).__init__()
            in_features = np.prod(input_shape[-2:])
            self.latent_dim = latent_dim
            self.num_behaviour = num_behaviour
            self.tau = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 50),
                nn.ReLU(),
                nn.Linear(50, 30),
                nn.ReLU(),
                nn.Linear(30, 25),
                nn.ReLU(),
                nn.Linear(25, 10),
                nn.ReLU(),
                nn.Linear(10, latent_dim),
                nn.BatchNorm1d(latent_dim),
                GaussianNoise(gaussian_noise),
            )
            self.T_Y = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
            )
            self.predictor = nn.Sequential(
                nn.Linear(latent_dim, num_behaviour),
            )

        def forward(self, x):
            # Upper arm of commutativity diagram
            yt1_upper = self.tau(x[:, 1])
            bt1_upper = self.predictor(yt1_upper)

            # Lower arm of commutativity diagram
            yt_lower = self.tau(x[:, 0])
            yt1_lower = yt_lower + self.T_Y(yt_lower)

            return yt1_upper, yt1_lower, bt1_upper

        def get_config(self):
            config = {
                'latent_dim': self.latent_dim,
                'num_behaviour': self.num_behaviour,
            }
            base_config = super(BunDLeNet, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

        @classmethod
        def from_config(cls, config):
            return cls(
                latent_dim=config['latent_dim'],
                num_behaviour=config['num_behaviour'],
            )


    # deploy BunDLe Net
    model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names), input_shape=x_.shape)
    loss_array, _ = train_model(
        x_,
        b_,
        model,
        b_type='discrete',
        gamma=0.9,
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
    vis.plot_latent_timeseries()
    vis.plot_phase_space()
    # vis.rotating_plot(
    #     filename=f'c_elegans_results/ablation_studies/figures/rotation_{algorithm}_worm_{worm_num}_gamma_{gamma}.gif',
    #     show_fig=False,
    #     #arrow_length_ratio=0.01
    #     )

plt.show()