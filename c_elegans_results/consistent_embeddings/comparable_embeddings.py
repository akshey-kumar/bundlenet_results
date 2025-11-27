import os
import torch
import numpy as np
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data

'''
Layers of the pretrained model on worm_num_i are  
used in learning a comparable embedding on worm_no_j.
so data corresponds to worm_no_j
model_old corresponds to worm_no_i
model_new will be trained on worm_no_j data
'''

worm_num_i = 0


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

# Load old pretrained model created in first_embedding.py (of worm_no_i)
## Load old data (reqiured to build old model)
model_path = f'data/generated/pytorch_models/c_elegans/bundle_net_model_worm_{worm_num_i}.pt'
from torch.serialization import safe_globals
with safe_globals([BunDLeNet]):
    model_old = torch.load(model_path, weights_only=False)
model_old.eval()

weights_T_Y = model_old.T_Y.state_dict()
weights_predictor = model_old.predictor.state_dict()

# Load new data and new model (of worm_no_j)
for worm_num_j in range(5):
    data = Database(data_path=data_path, dataset_no=worm_num_j)
    data.exclude_neurons(b_neurons)
    X = data.neuron_traces.T
    B = data.behaviour
    X_, B_ = prep_data(X, B, win=15)

    # Initialise new model and copy weights
    model_new = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names), input_shape=X_.shape)
    model_new.T_Y.load_state_dict(weights_T_Y)
    model_new.predictor.load_state_dict(weights_predictor)

    # Freeze T_Y and predictor, keep tau trainable
    for param in model_new.T_Y.parameters():
        param.requires_grad = False
    for param in model_new.predictor.parameters():
        param.requires_grad = False
    for param in model_new.tau.parameters():
        param.requires_grad = True

    # Training loop
    loss_array, _ = train_model(
        X_,
        B_,
        model_new,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=2000,
        initialisation=None,
    )

    # Projecting into latent space
    Y0_ = project_into_latent_space(X_[:, 0], model_new)

    algorithm = 'bundlenet'
    os.makedirs(f'data/generated/embeddings/c_elegans/comparable_embeddings/', exist_ok=True)
    np.savetxt(f'data/generated/embeddings/c_elegans/comparable_embeddings/Y0__{algorithm}_worm_{worm_num_j}', Y0_)
    np.savetxt(f'data/generated/embeddings/c_elegans/comparable_embeddings/B__{algorithm}_worm_{worm_num_j}', B_)
