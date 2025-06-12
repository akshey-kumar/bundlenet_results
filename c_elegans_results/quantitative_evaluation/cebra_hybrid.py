import os
import numpy as np
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from cebra import CEBRA

# load data (excluding behavioural neurons) and plot
for worm_num in range(5):
    algorithm = 'cebra_h'
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
    x_, b_ = prep_data(x, b, win=1) # the time delay embedding is done through the time_offsets parameter in CEBRA

    # train test split
    x_train, x_test, b_train, b_test = timeseries_train_test_split(x_, b_)
    x0_tr = x_train[:, 0, :, :].reshape(x_train.shape[0], -1)
    x1_tr = x_train[:, 1, :, :].reshape(x_train.shape[0], -1)
    x0_tst = x_test[:, 0, :, :].reshape(x_test.shape[0], -1)
    x1_tst = x_test[:, 1, :, :].reshape(x_test.shape[0], -1)

    '''
    Best hyperparameters found were:

    model_architecture: offset10-model
    batch_size: 256
    learning_rate: 6.553070915852317e-05
    temperature: 0.45593346212357455
    max_iterations: 5000
    distance: cosine
    time_offsets: 10
    '''
    # fit CEBRA hybrid
    cebra_hybrid_model = CEBRA(
        model_architecture='offset10-model',
        batch_size=256,
        learning_rate=6.553070915852317e-05,
        temperature=0.45593346212357455,
        output_dimension=3,
        max_iterations=5000,
        distance='cosine',
        conditional='time_delta',
        device='cuda_if_available',
        verbose=True,
        time_offsets=10,
        hybrid=True
    )
    cebra_hybrid_model.fit(x1_tr, b_train.astype(float))
    print(worm_num)

    ### Projecting into latent space
    y0_tr = cebra_hybrid_model.transform(x0_tr)
    y1_tr = cebra_hybrid_model.transform(x1_tr)
    y0_tst = cebra_hybrid_model.transform(x0_tst)
    y1_tst = cebra_hybrid_model.transform(x1_tst)

    # save the weights
    save_model = True
    if save_model:
        os.makedirs('data/generated/quantitative_evaluation/embeddings/c_elegans/', exist_ok=True)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y0_tr__{algorithm}_worm_{worm_num}', y0_tr)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y1_tr__{algorithm}_worm_{worm_num}', y1_tr)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y0_tst__{algorithm}_worm_{worm_num}', y0_tst)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/y1_tst__{algorithm}_worm_{worm_num}', y1_tst)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/b_tr__{algorithm}_worm_{worm_num}', b_train)
        np.savetxt(f'data/generated/quantitative_evaluation/embeddings/c_elegans/b_tst__{algorithm}_worm_{worm_num}', b_test)