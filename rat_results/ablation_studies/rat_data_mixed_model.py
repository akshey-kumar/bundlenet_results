import numpy as np
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
import matplotlib.pyplot as plt

algorithm = 'BunDLeNet_mixed_model'
rat_name = 'achilles' #, 'gatsby','cicero', 'buddy'

data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
x, b = data['x'], data['b']
x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values
np.where(x < 0)
x_, b_ = prep_data(x, b, win=20)

# Train test split
x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)

for gamma in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 1]:
    # Deploy BunDLe Net
    model = BunDLeNet(latent_dim=3, num_behaviour=b_.shape[1])
    train_history, test_history = train_model(
        x_train,
        b_train_1,
        model,
        b_type='continuous',
        gamma=gamma,
        learning_rate=0.001,
        n_epochs=500,
        initialisation=None,
        validation_data=(x_test, b_test_1),
    )

    plt.figure()
    for i, label in enumerate([
        r"$\mathcal{L}_{\mathrm{Markov}}$",
        r"$\mathcal{L}_{\mathrm{Behavior}}$",
        r"Train loss $\mathcal{L}$"
    ]):
        plt.plot(train_history[:, i], label=label)
    plt.plot(test_history[:, -1], label='Test loss')
    plt.title(f'gamma={gamma}')
    plt.legend()

    # Projecting into latent space
    y0_ = model.tau(x_[:, 0]).numpy()

    # Save the weights
    save_model = True
    if save_model:
        model.save_weights(f'data/generated/BunDLeNet_model_rat_{rat_name}_gamma_{gamma}')
        np.savetxt(f'data/generated/saved_Y/y0__{algorithm}_rat_{rat_name}_gamma_{gamma}', y0_)
        np.savetxt(f'data/generated/saved_Y/b__{algorithm}_rat_{rat_name}_gamma_{gamma}', b_)
        y0_ = np.loadtxt(f'data/generated/saved_Y/y0__{algorithm}_rat_{rat_name}_gamma_{gamma}')
        b_ = np.loadtxt(f'data/generated/saved_Y/b__{algorithm}_rat_{rat_name}_gamma_{gamma}').astype(int)

    y0_ = np.loadtxt(f'data/generated/saved_Y/y0__{algorithm}_rat_{rat_name}_gamma_{gamma}')
    b_ = np.loadtxt(f'data/generated/saved_Y/b__{algorithm}_rat_{rat_name}_gamma_{gamma}').astype(int)

    # Plotting latent space dynamics
    vis = LatentSpaceVisualiser(y0_, b_, data.behaviour_names, show_points=True)
    # vis.plot_latent_timeseries()
    # vis.plot_phase_space()
    vis.rotating_plot(filename=f'figures/rotation_{algorithm}_rat_{rat_name}_gamma_{gamma}.gif', show_fig=False,
                      arrow_length_ratio=0.01)
plt.show()