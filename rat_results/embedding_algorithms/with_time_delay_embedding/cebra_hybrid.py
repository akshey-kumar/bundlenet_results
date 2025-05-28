import numpy as np
from cebra import CEBRA
from ncmcm.bundlenet.utils import prep_data


algorithm = 'cebra_hybrid'
for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    # Load data
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values

    # time delay embedding
    x_, b_ = prep_data(x, b, win=1)

    # five fits of the model and pick the best model
    best_model = None
    lowest_loss = float("inf")
    for _ in range(5):
        # fit CEBRA hybrid - Optimal hyperparameters for this dataset are taken from CEBRA own demo notebook
        cebra_hybrid_model = CEBRA(
            model_architecture='offset10-model',
            batch_size=512,
            learning_rate=3e-4,
            temperature=1,
            output_dimension=3,
            max_iterations=5000,
            distance='cosine',
            conditional='time_delta',
            device='cuda_if_available',
            verbose=True,
            time_offsets=10,
            hybrid=True
        )
        cebra_hybrid_model.fit(x_[:, 0, 0, :], b_.astype(float))

        loss = cebra_hybrid_model.state_dict_["loss"][-1].item()
        if loss < lowest_loss:
            best_model, lowest_loss = cebra_hybrid_model, loss

    # projecting into latent space
    y_ = best_model.transform(x_[:, 0, 0, :])

    # save the weights
    save_model = True
    if save_model:
        np.savetxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}', y_)
        np.savetxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}', b_)
        y_ = np.loadtxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}')
        b_ = np.loadtxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}').astype(int)
