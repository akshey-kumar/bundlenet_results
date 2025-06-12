import sys
sys.path.append(r'../')
import numpy as np
from functions import *

algorithm = sys.argv[1]
worm_num = sys.argv[2]
state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
# Plotting

Y0_tr = np.loadtxt('data/generated/saved_Y/Y0_tr__'+algorithm+'_worm_'+ str(worm_num))
Y1_tr = np.loadtxt('data/generated/saved_Y/Y1_tr__'+algorithm+'_worm_'+ str(worm_num))
Y0_tst = np.loadtxt('data/generated/saved_Y/Y0_tst__'+algorithm+'_worm_'+ str(worm_num))
Y1_tst = np.loadtxt('data/generated/saved_Y/Y1_tst__'+algorithm+'_worm_'+ str(worm_num))
B_train_1 = np.loadtxt('data/generated/saved_Y/B_train_1__'+algorithm+'_worm_'+ str(worm_num)).astype(int)
B_test_1 = np.loadtxt('data/generated/saved_Y/B_test_1__'+algorithm+'_worm_'+ str(worm_num)).astype(int)


fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
plot_ps_(fig, ax, Y=Y0_tr, B=B_train_1, state_names=state_names, show_points=False)
plot_ps_(fig, ax, Y=Y0_tst, B=B_test_1, state_names=state_names, show_points=True)

plt.show()
