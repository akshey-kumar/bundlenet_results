import numpy as np
from ncmcm.data_loaders.matlab_dataset import Database

algorithm = 'PCA'

for worm_num in range(5):
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
    np.save(f'data/raw/c_elegans/neural_data_worm_{worm_num}', x)
    np.save(f'data/raw/c_elegans/behaviour_data_worm_{worm_num}.npy', b)