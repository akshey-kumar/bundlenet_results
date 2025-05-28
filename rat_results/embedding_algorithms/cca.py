import numpy as np
from sklearn.cross_decomposition import CCA
from ncmcm.bundlenet.utils import prep_data

algorithm = 'cca'
for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    # Load data
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values

    # time delay embedding
    x_, b_ = prep_data(x, b, win=1)
    x_ = x_[:,-1,:,:].reshape(x_.shape[0], -1)

    # fit CCA
    dim = 3
    cca = CCA(n_components=dim)
    cca.fit(x_, b_)
    print('Accuracy of CCA on data', cca.score(x_, b_))

    # projecting into latent space
    y_ = cca.transform(x_)

    # save the weights
    save_model = True
    if save_model:
        np.savetxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}', y_)
        np.savetxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}', b_)
        y_ = np.loadtxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}')
        b_ = np.loadtxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}').astype(int)
