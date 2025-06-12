import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import os

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

y0_tr = y0_tr.reshape(y0_tr.shape[0],-1)
y1_tr = y1_tr.reshape(y1_tr.shape[0],-1)
y0_tst = y0_tst.reshape(y0_tst.shape[0],-1)
y1_tst = y1_tst.reshape(y1_tst.shape[0],-1)
ydiff_tr = y1_tr - y0_tr
ydiff_tst = y1_tst - y0_tst

## behaviour decodability evaluation

# behavior predictor model
class BehaviorPredictor(nn.Module):
    def __init__(self):
        super(BehaviorPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(y1_tr.shape[1], 8)  # Input size matches y1_tr's feature dimension
        )

    def forward(self, x):
        return self.model(x)


# training and evaluation
acc_list = []
for _ in tqdm(range(10)):
    b_predictor = BehaviorPredictor()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(b_predictor.parameters(), lr=0.01)

    # training loop
    for epoch in range(100):
        b_predictor.train()
        optimizer.zero_grad()
        outputs = b_predictor(y1_tr)
        loss = criterion(outputs, b_tr)
        loss.backward()
        optimizer.step()

    # evaluation
    b_predictor.eval()
    with torch.no_grad():
        b1_tst_pred = b_predictor(y1_tst).argmax(dim=1).numpy()
        acc_list.append(metrics.accuracy_score(b1_tst_pred, b_tst.numpy()))

# save metrics
acc_list = np.array(acc_list)
os.makedirs('data/generated/c_elegans_evaluation_metrics', exist_ok=True)
np.savetxt(f'data/generated/c_elegans_evaluation_metrics/acc_list_{algorithm}_worm_{worm_num}', acc_list)
