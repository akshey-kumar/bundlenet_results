import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import timeseries_train_test_split, prep_data
from interpolation import interpolate_bouts
from time_alignment import extract_bouts
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

algorithm = 'BunDLeNet'
worm_num = 0

# loading neuronal data
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
x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)

x0_tr = x_train[:, 0, -1, :]
x0_tst = x_test[:, 0, -1, :]
x1_tr = x_train[:, 1, -1, :]
x1_tst = x_test[:, 1, -1, :]

bout_indices, next_b, _ = extract_bouts(b_train_1, b=7)  # anaylsing the braiching points of behaviours 5 (sus rev) and 7 (ventral turn)
x_bouts = [x0_tr[idx] for idx in bout_indices]
x_bouts = interpolate_bouts(x_bouts, t_steps_interp=20, show_plot=True)


'''
# loading embedding
file_pattern = f'data/generated/saved_Y/{{}}__{algorithm}_worm_{worm_num}'
Y0_tr = np.loadtxt(file_pattern.format('Y0_tr'))
Y1_tr = np.loadtxt(file_pattern.format('Y1_tr'))
Y0_tst = np.loadtxt(file_pattern.format('Y0_tst'))
Y1_tst = np.loadtxt(file_pattern.format('Y1_tst'))
B_train_1 = np.loadtxt(file_pattern.format('B_train_1')).astype(int)
B_test_1 = np.loadtxt(file_pattern.format('B_test_1')).astype(int)

bout_indices, next_b, _ = extract_bouts(B_train_1, b=5)
Y_bouts = [Y0_tr[idx] for idx in bout_indices]
Y_bouts = interpolate_bouts(Y_bouts, t_steps_interp=20, show_plot=True)
'''
# Define the classifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

clf = make_pipeline(
    PCA(n_components=3),
    LogisticRegression()
)
#clf = LinearSVC(C=0.35, penalty='l2')
#clf = LogisticRegression()
n_folds = 5
scores = np.zeros((x_bouts.shape[2], n_folds))


for t in range(x_bouts.shape[2]):
    x_bout_t = x_bouts[:, :, t]
    scores[t] = cross_val_score(clf, x_bout_t, next_b, cv=n_folds, scoring='accuracy')

mean_scores = scores.mean(axis=1)
std_error = scores.std(axis=1) / np.sqrt(n_folds)  # Standard error of the mean

clf.fit(x_bout_t, next_b)
print('train_score', clf.score(x_bout_t, next_b))

# Plot mean accuracy with error bars
plt.figure()
plt.plot(np.linspace(0, 1, 20), mean_scores)
plt.fill_between(np.linspace(0, 1, 20),
                 mean_scores - std_error,
                 mean_scores + std_error,
                 color="b", alpha=0.2)
plt.ylim([0.5, 1])
plt.xlabel("normalised time")
plt.ylabel("decoding accuracy")
plt.grid(True)

plt.show()
