import numpy as np
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from sklearn.metrics import mean_squared_error
import scipy


class ReducedRankRegressor(object):
    """
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - rrank is a rank constraint.
    - reg is a regularization parameter (optional).
    """
    def __init__(self, X, Y, rank, reg=None):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))
        if reg is None:
            reg = 0
        self.rank = rank

        CXX = np.dot(X.T, X) + reg * scipy.sparse.eye(np.size(X, 1))
        CXY = np.dot(X.T, Y)
        _U, _S, V = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))
        self.W = V[0:rank, :].T
        self.A = np.dot(np.linalg.pinv(CXX), np.dot(CXY, self.W)).T

    def __str__(self):
        return 'Reduced Rank Regressor (rank = {})'.format(self.rank)

    def project(self, X):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return np.asarray(np.dot(X, self.A.T))

    def predict(self, X):
        """Predict Y from X."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return np.asarray(np.dot(X, np.dot(self.A.T, self.W.T)))


algorithm = 'rrr_tde'

for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    # Load data
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values
    np.where(x < 0)
    x_, b_ = prep_data(x, b, win=20)
    x_ = x_[:, -1, :, :].reshape(x_.shape[0], -1)

    ### Deploy RRR
    dim = 3
    rrr = ReducedRankRegressor(x_, b_, dim)

    ### Projecting into latent space
    y_ = rrr.project(x_)

    # Predicting
    b_pred = rrr.predict(x_)
    print('mse', mean_squared_error(b_pred, b_))

    # save the weights
    save_model = True
    if save_model:
        np.savetxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}', y_)
        np.savetxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}', b_)
        y_ = np.loadtxt(f'data/generated/embeddings/rat/y0__{algorithm}_rat_{rat_name}')
        b_ = np.loadtxt(f'data/generated/embeddings/rat/b__{algorithm}_rat_{rat_name}').astype(int)
