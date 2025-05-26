import matplotlib.pyplot as plt
import numpy as np
# from sklearn.decomposition import PCA
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from cebra import CEBRA

# load Data (excluding behavioural neurons) and plot
for worm_num in range(5):
    algorithm = 'cebra_hybrid'
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

    # prepare data
    x_, b_ = prep_data(x, b, win=1)

    # five fits of the model and pick the best model
    best_model = None
    lowest_loss = float("inf")
    for _ in range(5):
        # fit CEBRA hybrid
        cebra_hybrid_model = CEBRA(
            model_architecture='offset10-model',
            batch_size=512,
            learning_rate=3e-4,
            temperature=1,
            output_dimension=3,
            max_iterations=15,#5000,
            distance='cosine',
            conditional='time_delta',
            device='cuda_if_available',
            verbose=True,
            time_offsets=10,
            hybrid = True
        )
        cebra_hybrid_model.fit(x_[:, 0, 0, :], b_.astype(float))

        loss = cebra_hybrid_model.state_dict_["loss"][-1].item()
        if loss < lowest_loss:
            best_model, lowest_loss = cebra_hybrid_model, loss

    # projecting into latent space
    y0_ = best_model.transform(x_[:, 0, 0, :])

    # save the weights
    save_model = True
    if save_model:
        np.savetxt(f'data/generated/embeddings/y0__{algorithm}_worm_{worm_num}', y0_)
        np.savetxt(f'data/generated/embeddings/b__{algorithm}_worm_{worm_num}', b_)
        y0_ = np.loadtxt(f'data/generated/embeddings/y0__{algorithm}_worm_{worm_num}')
        b_ = np.loadtxt(f'data/generated/embeddings/b__{algorithm}_worm_{worm_num}').astype(int)

    # plotting latent space dynamics
    #vis = LatentSpaceVisualiser(y0_, b_, data.behaviour_names)
    #vis.plot_latent_timeseries()
    #vis.plot_phase_space()
    #vis.rotating_plot(filename='figures/rotation_' + algorithm + '_worm_' + str(worm_num) + '.gif')
