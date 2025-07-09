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

file_pattern = f'data/generated/quantitative_evaluation/embeddings/c_elegans/{{}}__{algorithm}_worm_{worm_num}'
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

    # Evaluation
    mse_y1_y1_pred_tr = metrics.mean_squared_error(y1_tr.numpy(), y1_tr_pred.numpy())
    mse_y1_y1_pred_tst = metrics.mean_squared_error(y1_tst.numpy(), y1_tst_pred.numpy())

    mse_y1_y0_tr = metrics.mean_squared_error(y1_tr.numpy(), y0_tr.numpy())
    mse_y1_y0_tst = metrics.mean_squared_error(y1_tst.numpy(), y0_tst.numpy())

    mse_y1_y1_mean_tr = metrics.mean_squared_error(y1_tr.numpy(), np.full_like(y1_tr.numpy(), y1_tr.numpy().mean(axis=0)))
    mse_y1_y1_mean_tst = metrics.mean_squared_error(y1_tst.numpy(), np.full_like(y1_tst.numpy(), y1_tst.numpy().mean(axis=0)))

    mse_list.append([
        mse_y1_y1_pred_tr,
        mse_y1_y1_pred_tst,
        mse_y1_y0_tr,
        mse_y1_y0_tst,
        mse_y1_y1_mean_tr,
        mse_y1_y1_mean_tst
    ])
    r2_y1_y1_pred_tr = metrics.r2_score(y1_tr.numpy(), y1_tr_pred.numpy())
    r2_y1_y1_pred_tst = metrics.r2_score(y1_tst.numpy(), y1_tst_pred.numpy())

    r2_y1_y0_tr = metrics.r2_score(y1_tr.numpy(), y0_tr.numpy())
    r2_y1_y0_tst = metrics.r2_score(y1_tst.numpy(), y0_tst.numpy())

    r2_list.append([
        r2_y1_y1_pred_tr,
        r2_y1_y1_pred_tst,
        r2_y1_y0_tr,
        r2_y1_y0_tst
    ])

# Saving the metrics
mse_list = np.array(mse_list)
r2_list = np.array(r2_list)
os.makedirs('data/generated/quantitative_evaluation/evaluation_metrics/c_elegans', exist_ok=True)
np.savetxt(f'data/generated/quantitative_evaluation/evaluation_metrics/c_elegans/mse_list_{algorithm}_worm_{worm_num}', mse_list)
np.savetxt(f'data/generated/quantitative_evaluation/evaluation_metrics/c_elegans/r2_list_{algorithm}_worm_{worm_num}', r2_list)
