
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
from preprocess import preprocess_data
from torch.utils.data import TensorDataset, DataLoader

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
X1_tr, X1_tst, B_train_1, B_test_1 = timeseries_train_test_split(x_, b_)

# Flatten the inputs (same as tf.keras.layers.Flatten)
X1_tr_flat = X1_tr.reshape((X1_tr.shape[0], -1))
X1_tst_flat = X1_tst.reshape((X1_tst.shape[0], -1))

# Convert to torch tensors
x_train = torch.tensor(X1_tr_flat, dtype=torch.float32)
x_test = torch.tensor(X1_tst_flat, dtype=torch.float32)
y_train = torch.tensor(B_train_1, dtype=torch.long)
y_test = torch.tensor(B_test_1, dtype=torch.long)

# DataLoader
batch_size = 100
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define model
class BehaviorPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 8)  # output logits for 8 classes

    def forward(self, x):
        return self.linear(x)

# Training and evaluation loop
acc_list = []
for _ in tqdm(range(5)):
    model = BehaviorPredictor(input_dim=x_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        pred_classes = test_logits.argmax(dim=1).numpy()
        acc = accuracy_score(pred_classes, B_test_1)
        acc_list.append(acc)

acc_list = np.array(acc_list)
print(" neuronal prediction accuracy:", acc_list.mean().round(3), "pm", acc_list.std().round(3))