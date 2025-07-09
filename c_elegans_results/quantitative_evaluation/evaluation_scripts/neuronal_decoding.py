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
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split

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
x_, b_ = prep_data(x, b, win=1)
x_tr, x_tst, b_tr, b_tst = timeseries_train_test_split(x_, b_)
x1_tr = x_tr[:, 1, :, :].reshape(x_tr.shape[0], -1)
x1_tst = x_tst[:, 1, :, :].reshape(x_tst.shape[0], -1)

# Convert data to PyTorch tensors
x1_tr = torch.tensor(x1_tr, dtype=torch.float32)
x1_tst = torch.tensor(x1_tst, dtype=torch.float32)
b_tr = torch.tensor(b_tr, dtype=torch.long)
b_tst = torch.tensor(b_tst, dtype=torch.long)

## behaviour decodability evaluation
# behavior predictor model
class BehaviorPredictor(nn.Module):
    def __init__(self):
        super(BehaviorPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(x1_tr.shape[1], 8)
        )

    def forward(self, x):
        return self.model(x)

# training and evaluation
acc_list = []
for _ in tqdm(range(50)):
    b_predictor = BehaviorPredictor()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(b_predictor.parameters(), lr=0.01)

    # training loop
    for epoch in range(100):
        b_predictor.train()
        optimizer.zero_grad()
        outputs = b_predictor(x1_tr)
        loss = criterion(outputs, b_tr)
        loss.backward()
        optimizer.step()

    # evaluation
    b_predictor.eval()
    with torch.no_grad():
        b1_tst_pred = b_predictor(x1_tst).argmax(dim=1).numpy()
        acc_list.append(metrics.accuracy_score(b1_tst_pred, b_tst.numpy()))

# save metrics
acc_list = np.array(acc_list)
os.makedirs('data/generated/quantitative_evaluation/evaluation_metrics/c_elegans', exist_ok=True)
np.savetxt(f'data/generated/quantitative_evaluation/evaluation_metrics/c_elegans/acc_list_X_worm_{worm_num}', acc_list)



### Estimating the chance accuracy of behaviour decoding
chance_acc = np.zeros(500)
for i, _ in enumerate(chance_acc):
    b_perm = np.random.choice(b_tst, size=b_tst.shape)
    chance_acc[i] = accuracy_score(b_perm, b_tst)
print('Chance prediction accuracy: ', chance_acc.mean().round(3), ' pm ', chance_acc.std().round(3))
np.savetxt(f'data/generated/quantitative_evaluation/evaluation_metrics/c_elegans/acc_list_chance_worm_{worm_num}', chance_acc)
