import numpy as np
import matplotlib.pyplot as plt
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser


algorithm = 'bundlenet'
for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    # Load data
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values

    # time delay embedding
    x_, b_ = prep_data(x, b, win=20)

    # Deploy BunDLe Net
    model = BunDLeNet(latent_dim=3, num_behaviour=b_.shape[1], input_shape=x_.shape)

    loss_array, _ = train_model(
        x_,
        b_,
        model,
        b_type='continuous',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=500,
        initialisation='best_of_5_init',
    )

    for i, label in enumerate([
        r"$\mathcal{L}_{\mathrm{Markov}}$",
        r"$\mathcal{L}_{\mathrm{Behavior}}$",
        r"Total loss $\mathcal{L}$"
    ]):
        plt.plot(loss_array[:, i], label=label)
    plt.legend()
    # plt.show()

    # projecting into latent space
    y_ = project_into_latent_space(x_, model)

    # save the weights
    save_model = True
    if save_model:
        np.savetxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}', y_)
        np.savetxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}', b_)
        y_ = np.loadtxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}')
        b_ = np.loadtxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}').astype(int)
