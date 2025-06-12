import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import os
import sys

algorithm = sys.argv[1]
worm_num = int(sys.argv[2])
print(algorithm, ' worm_num: ', worm_num)

file_pattern = f'data/generated/saved_Y/{{}}__{algorithm}_worm_{worm_num}'
y0_tr = np.loadtxt(file_pattern.format('y0_tr'))
y1_tr = np.loadtxt(file_pattern.format('y1_tr'))
y0_tst = np.loadtxt(file_pattern.format('y0_tst'))
y1_tst = np.loadtxt(file_pattern.format('y1_tst'))
b_tr = np.loadtxt(file_pattern.format('b_tr'))
b_tst = np.loadtxt(file_pattern.format('b_tst'))

y0_tr = y0_tr.reshape(y0_tr.shape[0], -1)
y1_tr = y1_tr.reshape(y1_tr.shape[0], -1)
y0_tst = y0_tst.reshape(y0_tst.shape[0], -1)
y1_tst = y1_tst.reshape(y1_tst.shape[0], -1)
ydiff_tr = y1_tr - y0_tr
ydiff_tst = y1_tst - y0_tst

# Convert data to PyTorch tensors
y0_tr = torch.tensor(y0_tr, dtype=torch.float32)
y1_tr = torch.tensor(y1_tr, dtype=torch.float32)
y0_tst = torch.tensor(y0_tst, dtype=torch.float32)
y1_tst = torch.tensor(y1_tst, dtype=torch.float32)
ydiff_tr = torch.tensor(ydiff_tr, dtype=torch.float32)
ydiff_tst = torch.tensor(ydiff_tst, dtype=torch.float32)

# Dynamics predictability evaluation
mse_list = []
r2_list = []
for i in tqdm(range(10)):
    # Scaling input and output data
    yinmax = torch.abs(y0_tr).max()  # Parameters for scaling
    y0_tr, y0_tst = y0_tr / yinmax, y0_tst / yinmax
    ydmax = torch.abs(ydiff_tr).max()  # Parameters for scaling
    ydiff_tr, ydiff_tst = ydiff_tr / ydmax, ydiff_tst / ydmax

    # Defining the model
    model_ydiff_f_yt = nn.Sequential(
        nn.Linear(y0_tr.shape[1], 3)  # Input size matches y0_tr's feature dimension
    )
    optimizer = optim.Adam(model_ydiff_f_yt.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(50):
        model_ydiff_f_yt.train()
        optimizer.zero_grad()
        outputs = model_ydiff_f_yt(y0_tr)
        loss = criterion(outputs, ydiff_tr)
        loss.backward()
        optimizer.step()

    # Predictions
    model_ydiff_f_yt.eval()
    with torch.no_grad():
        ydiff_tr_pred = model_ydiff_f_yt(y0_tr)
        ydiff_tst_pred = model_ydiff_f_yt(y0_tst)

    # Inverse scaling the data
    ydiff_tr_pred, ydiff_tr, y0_tr = ydiff_tr_pred * ydmax, ydiff_tr * ydmax, y0_tr * yinmax
    ydiff_tst_pred, ydiff_tst, y0_tst = ydiff_tst_pred * ydmax, ydiff_tst * ydmax, y0_tst * yinmax

    y1_tr_pred = y0_tr + ydiff_tr_pred
    y1_tst_pred = y0_tst + ydiff_tst_pred

    # evaluation
    flat_partial = lambda x: x.reshape(x.shape[0], -1)
    mse_baseline_tr = metrics.mean_squared_error(flat_partial(y1_tr), flat_partial(y0_tr))
    mse_model_tr = metrics.mean_squared_error(flat_partial(y1_tr), flat_partial(y1_tr_pred))
    mse_baseline_tst = metrics.mean_squared_error(flat_partial(y1_tst), flat_partial(y0_tst))
    mse_model_tst = metrics.mean_squared_error(flat_partial(y1_tst), flat_partial(y1_tst_pred))

    mse_list.append([baseline_tr, modelmse_tr, baseline_tst, modelmse_tst])

    r2_baseline_tr = metrics.r2_score(flat_partial(y1_tr), flat_partial(y0_tr))
    r2_model_tr = metrics.r2_score(flat_partial(y1_tr), flat_partial(y1_tr_pred))
    r2_baseline_tst = metrics.r2_score(flat_partial(y1_tst), flat_partial(y0_tst))
    r2_model_tst = metrics.r2_score(flat_partial(y1_tst), flat_partial(y1_tst_pred))

    r2_list.append([r2_baseline_tr, r2_model_tr, r2_baseline_tst, r2_model_tst])

# Saving the metrics
mse_list = np.array(mse_list)
r2_list = np.array(r2_list)
os.makedirs('data/generated/c_elegans_evaluation_metrics', exist_ok=True)
np.savetxt(f'data/generated/c_elegans_evaluation_metrics/mse_list_{algorithm}_worm_{worm_num}', mse_list)
np.savetxt(f'data/generated/c_elegans_evaluation_metrics/r2_list_{algorithm}_worm_{worm_num}', r2_list)
