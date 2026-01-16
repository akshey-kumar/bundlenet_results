import os
import numpy as np
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
import matplotlib.pyplot as plt

algorithm = 'BunDLeNet_mixed_model'
rat_names = [ 'achilles' , 'gatsby','cicero', 'buddy']
for rat_name in rat_names:

    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)
    x_, b_ = prep_data(x, b, win=20)

    # Train test split
    x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)

    for gamma in [0,0.9,1]: #[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 1]:
        # Deploy BunDLe Net
        model = BunDLeNet(latent_dim=3, num_behaviour=b_.shape[1], input_shape=x_.shape)
        train_history, test_history = train_model(
            x_train,
            b_train_1,
            model,
            b_type='continuous',
            gamma=gamma,
            learning_rate=0.001,
            n_epochs=500,
            initialisation='best_of_5_init',
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
        y0_ = project_into_latent_space(x_[:, 0], model)
        # Save the weights
        save_model = True
        if save_model:
            os.makedirs('data/generated/embeddings/rat/ablation_experiments/', exist_ok=True)
            np.savetxt(f'data/generated/embeddings/rat/ablation_experiments/y0__{algorithm}_rat_{rat_name}_gamma_{gamma}', y0_)
            np.savetxt(f'data/generated/embeddings/rat/ablation_experiments/b__{algorithm}_rat_{rat_name}_gamma_{gamma}', b_)

        # Plotting latent space dynamics
        # Continuous variable plotting
        # fig = plt.figure(figsize=(4, 4))
        # ax = plt.axes(projection='3d')
        # ax.axis('off')
        # pts = ax.scatter(y0_[:, 0], y0_[:, 1], y0_[:, 2], c=b_[:, 0], s=0.5)
        # plt.colorbar(pts)

        # Discrete variable plotting
        # vis = LatentSpaceVisualiser(
        #     y=y0_,
        #     b=b_[:, 1].astype(int),
        #     b_names=['0', '1']
        # )
        # vis.plot_phase_space(
        #    show_fig=False,
        #    arrow_length_ratio=0.001,
        # )