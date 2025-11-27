import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import accuracy_score
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import timeseries_train_test_split
from torch.utils.data import TensorDataset, DataLoader
from preprocess import preprocess_data,  prep_data

worm_num = int(sys.argv[1])
print('worm_num: ', worm_num)

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

# prepare data for BunDLe Net
_, x = preprocess_data(x, float(data.fps))
x_, b_ = prep_data(x, b, win=15)
x_tr, x_tst, b_tr, b_tst = timeseries_train_test_split(x_, b_)
x1_tr = x_tr[:, 1, :, :].reshape(x_tr.shape[0], -1)
x1_tst = x_tst[:, 1, :, :].reshape(x_tst.shape[0], -1)

# Convert data to PyTorch tensors
x1_tr = torch.tensor(x1_tr, dtype=torch.float32)
x1_tst = torch.tensor(x1_tst, dtype=torch.float32)
b_tr = torch.tensor(b_tr, dtype=torch.long)
b_tst = torch.tensor(b_tst, dtype=torch.long)



# Create DataLoader with batch size 100
batch_size = 100
train_dataset = TensorDataset(x1_tr, b_tr)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# behavior predictor model
class BehaviorPredictor(nn.Module):
    def __init__(self):
        super(BehaviorPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(x1_tr.shape[1], 8)
        )

    def forward(self, x):
        return self.model(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)



# training and evaluation
acc_list = []
for _ in tqdm(range(5)):
    b_predictor = BehaviorPredictor()
    b_predictor.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(b_predictor.parameters(), lr=0.01)

    # training loop
    for epoch in range(100):
        b_predictor.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = b_predictor(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

    # evaluation
    b_predictor.eval()
    with torch.no_grad():
        b1_tst_pred = b_predictor(x1_tst).argmax(dim=1).numpy()
        acc_list.append(metrics.accuracy_score(b1_tst_pred, b_tst.numpy()))

# save metrics
acc_list = np.array(acc_list)
print(' neuronal prediction accuracy: ', acc_list.mean().round(3), ' pm ', acc_list.std().round(3))

os.makedirs('data/generated/quantitative_evaluation/evaluation_metrics/c_elegans', exist_ok=True)
np.savetxt(f'data/generated/quantitative_evaluation/evaluation_metrics/c_elegans/acc_list_X_worm_{worm_num}', acc_list)



### Estimating the chance accuracy of behaviour decoding
chance_acc = np.zeros(500)
for i, _ in enumerate(chance_acc):
    b_perm = np.random.choice(b_tst, size=b_tst.shape)
    chance_acc[i] = accuracy_score(b_perm, b_tst)
print('Chance prediction accuracy: ', chance_acc.mean().round(3), ' pm ', chance_acc.std().round(3))
np.savetxt(f'data/generated/quantitative_evaluation/evaluation_metrics/c_elegans/acc_list_chance_worm_{worm_num}', chance_acc)
