import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from inverse_embedding import fit_inverse_embedder
from time_alignment import extract_bouts
from interpolation import interpolate_bouts, plot_bouts

# Loading data
algorithm = 'BunDLeNet'
worm_num = 0
data_path = 'data/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
b_neurons = [
    'AVAR',
    'AVAL',
    'SMDVR',
    'SMDVL',
    'SMDDR',
    'SMDDL',
    'RIBR',
    'RIBL', ]
data.exclude_neurons(b_neurons)
x = data.neuron_traces.T
b = data.behaviour
x_, b_ = prep_data(x, b, win=15)
x_train, x_test, _, _ = timeseries_train_test_split(x_, b_)
x0_tr = x_train[:, 0, -1, :]
x0_tst = x_test[:, 0, -1, :]

file_pattern = f'data/generated/saved_Y/{{}}__{algorithm}_worm_{worm_num}'
y0_tr = np.loadtxt(file_pattern.format('Y0_tr'))
y0_tst = np.loadtxt(file_pattern.format('Y0_tst'))
b_train_1 = np.loadtxt(file_pattern.format('B_train_1')).astype(int)
b_test_1 = np.loadtxt(file_pattern.format('B_test_1')).astype(int)

# fitting inverse embedder
inverse_embedder = fit_inverse_embedder(x0_tr, y0_tr, x0_tst, y0_tst)

'''
# points to inverse embed(ventral turn)
bout_indices, next_b, prev_b = extract_bouts(b_train_1, 7)
print(next_b)
print(prev_b)

y_bouts = [y0_tr[idx] for idx in bout_indices]
y_bouts = interpolate_bouts(y_bouts, t_steps_interp=21)
for i in range(8):
    print(i, y_bouts[prev_b==i].shape)


y0_inv_embed = []
y0_inv_embed.append(y_bouts[:,:,0].mean(axis=0))
y0_inv_embed.append(y_bouts[:,:,10].mean(axis=0))
y0_inv_embed.append(y_bouts[next_b==1][:,:,20].mean(axis=0))
y0_inv_embed.append(y_bouts[next_b==6][:,:,20].mean(axis=0))

y0_inv_embed = np.array(y0_inv_embed)
print(y0_inv_embed)

vis = LatentSpaceVisualiser(y0_tr, b_train_1, data.behaviour_names)
fig, ax = vis.plot_phase_space(axis_view=(0,0,),  arrow_length_ratio=0.2, show_fig=False)
ax.scatter(y0_inv_embed[:, 0], y0_inv_embed[:, 1], y0_inv_embed[:, 2], c='k', s=25, marker='x')
plt.show()
'''


# points to inverse embed (sustained reversal)
bout_indices, next_b, prev_b = extract_bouts(b_train_1, 5)
y_bouts = [y0_tr[idx] for idx in bout_indices]
y_bouts = interpolate_bouts(y_bouts, t_steps_interp=21)
for i in range(8):
    print(i, y_bouts[next_b==i].shape)


y0_inv_embed = []

y0_inv_embed.append(y_bouts[prev_b==3][:,:,0].mean(axis=0))
y0_inv_embed.append(y_bouts[prev_b==4][:,:,0].mean(axis=0))
y0_inv_embed.append(y_bouts[:,:,10].mean(axis=0))
y0_inv_embed.append(y_bouts[next_b==0][:,:,20].mean(axis=0))
y0_inv_embed.append(y_bouts[next_b==7][:,:,20].mean(axis=0))
y0_inv_embed = np.array(y0_inv_embed)
print(y0_inv_embed)


vis = LatentSpaceVisualiser(y0_tr, b_train_1, data.behaviour_names)
fig, ax = vis.plot_phase_space(axis_view=(0,0,),  arrow_length_ratio=0.2, show_fig=False)
ax.scatter(y0_inv_embed[:, 0], y0_inv_embed[:, 1], y0_inv_embed[:, 2], c='k', s=55, marker='x')
plt.show()


x0_pred = inverse_embedder(y0_inv_embed).numpy()
print(x0_pred.shape)
print(x0_pred)
for i, x_i in enumerate(x0_pred):
    plt.figure(figsize=(5, 2.5))
    for n, x_n in enumerate(x_i):
        plt.ylim(0, 1.1*x0_pred.max())
        plt.scatter(n, x_n)
    plt.title(y0_inv_embed[i])

plt.show()

