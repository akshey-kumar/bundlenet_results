import sys
import numpy as np
import matplotlib.pyplot as plt

algorithm = sys.argv[1]
rat_name = sys.argv[2] #'achilles', 'gatsby', 'cicero', 'buddy'
print(algorithm, 'rat_name: ', rat_name)

file_pattern = f'data/generated/predicted_and_true_behaviours/{{}}__{algorithm}_rat_{rat_name}'
b_train_1_pred = np.loadtxt(file_pattern.format('b_train_1_pred'))
b_test_1_pred = np.loadtxt(file_pattern.format('b_test_1_pred'))
b_train_1 = np.loadtxt(file_pattern.format('b_train_1'))
b_test_1 = np.loadtxt(file_pattern.format('b_test_1'))

plt.figure(figsize=(10,3))
plt.plot(b_test_1_pred[:,0], label=f'{algorithm} predicted behaviour')
plt.plot(b_test_1[:,0], label='True behaviour')
plt.ylabel('position')
plt.xlabel('time')
plt.legend()

plt.figure(figsize=(10,3))
plt.plot(b_train_1_pred[:,0], label=f'{algorithm} predicted behaviour')
plt.plot(b_train_1[:,0], label='True behaviour')
plt.ylabel('position')
plt.xlabel('time')
plt.legend()
plt.show()