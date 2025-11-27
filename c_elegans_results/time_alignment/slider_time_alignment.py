import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ncmcm.data_loaders.matlab_dataset import Database

algorithm = 'BunDLeNet'
worm_num = 3
print(algorithm, ' worm_num: ', worm_num)

data_path = 'data/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)

file_pattern = f'data/generated/embeddings/c_elegans/{{}}__{algorithm}_worm_{worm_num}'
Y0_tr = np.loadtxt(file_pattern.format('Y0_tr'))
Y1_tr = np.loadtxt(file_pattern.format('Y1_tr'))
Y0_tst = np.loadtxt(file_pattern.format('Y0_tst'))
Y1_tst = np.loadtxt(file_pattern.format('Y1_tst'))
B_train_1 = np.loadtxt(file_pattern.format('B_train_1')).astype(int)
B_test_1 = np.loadtxt(file_pattern.format('B_test_1')).astype(int)

# Initial values for elevation and azimuth
elev_init = 0
azim_init = 0
"""
# Function to create a rotation matrix
def rotation_matrix(elev, azim):
    azim = np.deg2rad(azim)
    elev = np.deg2rad(elev)
    R_z = np.array([[np.cos(azim), np.sin(azim), 0],
                    [-np.sin(azim), np.cos(azim), 0],
                    [0, 0, 1]])
    R_y = np.array([[np.cos(elev), 0, np.sin(elev)],
                    [0, 1, 0],
                    [-np.sin(elev), 0, np.cos(elev)]])
    return R_y @ R_z


# Function to extract bouts based on behavior labels
def extract_bouts(B, b):
    bouts = []
    current_bout = []
    for i, val in enumerate(B):
        if val == b:
            current_bout.append(i)
        else:
            if current_bout:
                bouts.append(current_bout)
                current_bout = []
    if current_bout:
        bouts.append(current_bout)
    return sorted(bouts, key=len)


# Function to plot behaviors in normalized time
def plot_behaviours_in_normalised_time(ax, Y, B, b, behaviour_name):
    bout_indices = extract_bouts(B, b)
    Y_bouts = [Y[idx] for idx in bout_indices]
    colors = ["#008080", "#FF6F61", "#FFD700"]

    for bout_idx, y_bout in enumerate(Y_bouts):
        normalised_t = np.linspace(0, 1, y_bout.shape[0])
        for i in range(3):
            darkness_factor = 0.3 + 0.7 * (bout_idx / len(Y_bouts))
            line, = ax.plot(normalised_t, y_bout[:, i], c=colors[i], alpha=darkness_factor, marker='o')

    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel('Normalized Time')
    ax.set_yticks([])
    ax.set_title(f"Behaviour: {behaviour_name}")


# Function to rotate and plot behaviors
def rotate_plot_behaviours(ax, Y, B, b, behaviour_name, rot_mat):
    Y_rotated = (rot_mat @ Y.T).T
    return plot_behaviours_in_normalised_time(ax, Y_rotated, B, b, behaviour_name)


# Main plot setup
b = 7  # Behavior label to visualize
print(data.behaviour_names)
fig, ax = plt.subplots(figsize=(8, 7))
rot_mat_init = rotation_matrix(elev_init, azim_init)
rotate_plot_behaviours(ax, Y1_tr, B_train_1, b, data.behaviour_names[b], rot_mat_init)
plt.subplots_adjust(bottom=0.25)

# Slider setup
ax_elev = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_azim = plt.axes([0.2, 0.1, 0.65, 0.03])
slider_elev = Slider(ax_elev, 'Elevation', -90.0, 90.0, valinit=elev_init)
slider_azim = Slider(ax_azim, 'Azimuth', 0.0, 360.0, valinit=azim_init)

# Update function for sliders
def update(val):
    elev = slider_elev.val
    azim = slider_azim.val
    rot_mat = rotation_matrix(elev, azim)

    ax.cla()  # Clear the current axes
    rotate_plot_behaviours(ax, Y1_tr, B_train_1, b, data.behaviour_names[b], rot_mat)

    fig.canvas.draw_idle()  # Redraw the canvas


# Connect sliders to the update function
slider_elev.on_changed(update)
slider_azim.on_changed(update)
plt.show()


"""
def rotation_matrix(elev, azim):
    azim = np.deg2rad(azim)
    elev = np.deg2rad(elev)
    R_z = np.array([[np.cos(azim), np.sin(azim), 0],
                    [-np.sin(azim), np.cos(azim), 0],
                    [0, 0, 1]])
    R_y = np.array([[np.cos(elev), 0, np.sin(elev)],
                    [0, 1, 0],
                    [-np.sin(elev), 0, np.cos(elev)]])
    return R_y @ R_z


def extract_bouts(B, b):
    bouts = []
    current_bout = []
    for i, val in enumerate(B):
        if val == b:
            current_bout.append(i)
        else:
            if current_bout:
                bouts.append(current_bout)
                current_bout = []
    if current_bout:
        bouts.append(current_bout)
    return sorted(bouts, key=len)


def plot_behaviours_in_normalised_time(axs, Y, B, b, behaviour_name):
    bout_indices = extract_bouts(B, b)
    Y_bouts = [Y[idx] for idx in bout_indices]
    colors = ["#008080", "#FF6F61", "#FFD700"]

    for bout_idx, y_bout in enumerate(Y_bouts):
        normalised_t = np.linspace(0, 1, y_bout.shape[0])
        for i in range(3):
            darkness_factor = 0.3 + 0.7 * (bout_idx / len(Y_bouts))
            axs[i].plot(y_bout[:, i], normalised_t, c=colors[i], alpha=darkness_factor, marker='o', markersize=1)
            axs[i].set_xlim(Y[:, i].min(), Y[:, i].max())
            axs[i].set_xticks([])
            axs[i].set_title(f"{behaviour_name} (Dimension {i + 1})")
    axs[1].set_xlabel('Normalized Time')


def rotate_plot_behaviours(axs, Y, B, b, behaviour_name, rot_mat):
    Y_rotated = (rot_mat @ Y.T).T
    plot_behaviours_in_normalised_time(axs, Y_rotated, B, b, behaviour_name)


# Plot setup
b = 7
fig, axs = plt.subplots(1, 3, figsize=(8, 9))
rot_mat_init = rotation_matrix(elev_init, azim_init)
rotate_plot_behaviours(axs, Y1_tr, B_train_1, b, data.behaviour_names[b], rot_mat_init)
plt.subplots_adjust(bottom=0.25)

# Slider setup
ax_elev = plt.axes([0.2, 0.1, 0.65, 0.03])
ax_azim = plt.axes([0.2, 0.15, 0.65, 0.03])
slider_elev = Slider(ax_elev, 'Elevation', -90.0, 90.0, valinit=elev_init)
slider_azim = Slider(ax_azim, 'Azimuth', 0.0, 360.0, valinit=azim_init)


def update(val):
    elev = slider_elev.val
    azim = slider_azim.val
    rot_mat = rotation_matrix(elev, azim)

    for ax in axs:
        ax.cla()  # Clear each subplot
    rotate_plot_behaviours(axs, Y1_tr, B_train_1, b, data.behaviour_names[b], rot_mat)
    fig.canvas.draw_idle()


slider_elev.on_changed(update)
slider_azim.on_changed(update)

plt.show()
