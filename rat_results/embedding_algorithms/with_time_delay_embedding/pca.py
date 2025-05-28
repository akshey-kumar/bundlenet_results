import numpy as np
from sklearn.decomposition import PCA
from ncmcm.bundlenet.utils import prep_data

algorithm = 'pca_tde'
for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    # Load data
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values

    # time delay embedding
    x_, b_ = prep_data(x, b, win=20)
    x_ = x_[:, -1, :, :].reshape(x_.shape[0], -1)

    # fit PCA
    dim = 3
    pca = PCA(n_components=dim)
    pca.fit(x_)
    print('Percentage of variance explained by the first ', dim, ' PCs: ', pca.explained_variance_ratio_[:dim].sum().round(3))

    # projecting into latent space
    y_ = pca.transform(x_)

    # save the weights
    save_model = True
    if save_model:
        np.savetxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}', y_)
        np.savetxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}', b_)
        y_ = np.loadtxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}')
        b_ = np.loadtxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}').astype(int)
