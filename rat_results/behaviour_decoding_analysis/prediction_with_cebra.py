import os

import numpy as np
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
import matplotlib.pyplot as plt
from cebra import CEBRA

algorithm = 'cebra_hybrid'
for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values
    x_, b_ = prep_data(x, b, win=1)

    # Train test split
    x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)

    # Deploy CEBRA hybrid
    cebra_hybrid_model = CEBRA(model_architecture='offset10-model',
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
                               hybrid=True)

    cebra_hybrid_model.fit(x_train[:, 0, 0, :], b_train_1.astype(float))

    # Projecting into latent space
    y0_tr = cebra_hybrid_model.transform(x_train[:, 0, 0, :])
    y1_tr = cebra_hybrid_model.transform(x_train[:, 1, 0, :])
    y0_tst = cebra_hybrid_model.transform(x_test[:, 0, 0, :])
    y1_tst = cebra_hybrid_model.transform(x_test[:, 1, 0, :])

    # adapted from cebra demo notebook: Define decoding function with kNN decoder. For a simple demo, we will use the fixed number of neighbors 36.
    # https://github.com/AdaptiveMotorControlLab/CEBRA-demos/blob/main/Demo_hippocampus.ipynb
    def decoding_pos_dir(emb_train, emb_test, label_train, label_test, n_neighbors=36):
        from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
        pos_decoder = KNeighborsRegressor(n_neighbors, metric='cosine')
        dir_decoder_1 = KNeighborsClassifier(n_neighbors, metric='cosine')
        dir_decoder_2 = KNeighborsClassifier(n_neighbors, metric='cosine')

        pos_decoder.fit(emb_train, label_train[:, 0])
        dir_decoder_1.fit(emb_train, label_train[:, 1])
        dir_decoder_2.fit(emb_train, label_train[:, 2])

        b_train_1 = np.c_[pos_decoder.predict(emb_train), dir_decoder_1.predict(emb_train), dir_decoder_2.predict(emb_train)]
        b_test_1 = np.c_[pos_decoder.predict(emb_test), dir_decoder_1.predict(emb_test), dir_decoder_2.predict(emb_test)]

        print('b_train_1', b_train_1.shape)
        print('b_test_1', b_test_1.shape)
        return b_train_1, b_test_1


    b_train_1_pred, b_test_1_pred = decoding_pos_dir(y1_tr, y1_tst, b_train_1, b_test_1)
    
    # Save the weights
    os.makedirs('data/generated/predicted_and_true_behaviours', exist_ok=True)
    np.savetxt(f'data/generated/predicted_and_true_behaviours/b_train_1_pred__{algorithm}_rat_{rat_name}',
               b_train_1_pred)
    np.savetxt(f'data/generated/predicted_and_true_behaviours/b_test_1_pred__{algorithm}_rat_{rat_name}', b_test_1_pred)
    np.savetxt(f'data/generated/predicted_and_true_behaviours/b_train_1__{algorithm}_rat_{rat_name}', b_train_1)
    np.savetxt(f'data/generated/predicted_and_true_behaviours/b_test_1__{algorithm}_rat_{rat_name}', b_test_1)
    
