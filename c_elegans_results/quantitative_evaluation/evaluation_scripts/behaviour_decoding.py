import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

algorithm = sys.argv[1]
worm_num = int(sys.argv[2])
print(algorithm, ' worm_num: ', worm_num)
file_pattern = f'data/generated/quantitative_evaluation/embeddings/c_elegans/{{}}__{algorithm}_worm_{worm_num}'
y1_tr = np.loadtxt(file_pattern.format('y1_tr'))
y1_tst = np.loadtxt(file_pattern.format('y1_tst'))
b_tr = np.loadtxt(file_pattern.format('b_tr'))
b_tst = np.loadtxt(file_pattern.format('b_tst'))

'''
file_pattern = f'data/generated/old_saved_Y/{{}}__{algorithm}_worm_{worm_num}'
y1_tr = np.loadtxt(file_pattern.format('Y1_tr'))
y1_tst = np.loadtxt(file_pattern.format('Y1_tst'))
b_tr = np.loadtxt(file_pattern.format('B_train_1'))
b_tst = np.loadtxt(file_pattern.format('B_test_1'))
'''

y1_tr = y1_tr.reshape(y1_tr.shape[0],-1)
y1_tst = y1_tst.reshape(y1_tst.shape[0],-1)

# Convert data to PyTorch tensors
y1_tr = torch.tensor(y1_tr, dtype=torch.float32)
y1_tst = torch.tensor(y1_tst, dtype=torch.float32)
b_tr = torch.tensor(b_tr, dtype=torch.long)
b_tst = torch.tensor(b_tst, dtype=torch.long)

# Set up dataloader
batch_size = 100
train_ds = TensorDataset(y1_tr, b_tr)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

acc_list = []
for _ in tqdm(range(10)):
    # linear feedforward classifier
    model = nn.Sequential(nn.Linear(y1_tr.shape[1], 8))  # no activation (logits)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # training loop
    model.train()
    for epoch in range(100):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    # evaluate on test set
    model.eval()
    with torch.no_grad():
        logits = model(y1_tst)
        preds = logits.argmax(dim=1).cpu().numpy()
        acc = metrics.accuracy_score(b_tst.cpu().numpy(), preds)
        acc_list.append(acc)

# Print average accuracy
print(f"Average test accuracy over 10 runs: {np.mean(acc_list):.4f}")

# save metrics
acc_list = np.array(acc_list)
os.makedirs('data/generated/quantitative_evaluation/evaluation_metrics/c_elegans', exist_ok=True)
np.savetxt(f'data/generated/quantitative_evaluation/evaluation_metrics/c_elegans/acc_list_{algorithm}_worm_{worm_num}', acc_list)

'''
# Plot loss curves vs epochs

train_losses = []
test_losses = []
model = nn.Sequential(nn.Linear(y1_tr.shape[1], 8))  # no activation (logits)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# training loop
model.train()
for epoch in range(100):
    epoch_train_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_losses.append(epoch_train_loss / len(train_loader))

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(y1_tst)
        test_loss = criterion(test_logits, b_tst).item()
        test_losses.append(test_loss)


# evaluate on test set
model.eval()
with torch.no_grad():
    logits = model(y1_tst)
    preds = logits.argmax(dim=1).cpu().numpy()
    acc = metrics.accuracy_score(b_tst.cpu().numpy(), preds)
    print('test_acc', acc)

# Plot train and test loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), train_losses, label='Train Loss')
plt.plot(range(1, 101), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()
'''

