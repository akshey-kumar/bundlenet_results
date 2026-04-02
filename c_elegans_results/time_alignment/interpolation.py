import random
import numpy as np
import matplotlib.pyplot as plt

def interpolate_bouts(Y_bouts, t_steps_interp=20, show_plot=False):
    n_dims = Y_bouts[0].shape[-1]
    Y_bouts_interp = np.zeros((len(Y_bouts), n_dims, t_steps_interp))
    for bout_idx, y_bout in enumerate(Y_bouts):
        normalised_t = np.linspace(0, 1, y_bout.shape[0])
        interp_t = np.linspace(0, 1, t_steps_interp)

        colors = ["#008080", "#FF6F61", "#FFD700"]
        colors += [
            "#{:06x}".format(random.randint(0, 0xFFFFFF))
            for _ in range(300)
        ]
        for i in range(n_dims):
            darkness_factor = 0.3 + 0.7 * (bout_idx / len(Y_bouts))
            Y_bouts_interp[bout_idx, i, :] = np.interp(interp_t, normalised_t, y_bout[:,i])
            if show_plot:
                plt.plot(interp_t, Y_bouts_interp[bout_idx, i, :], c=colors[i], marker='o', alpha=darkness_factor)


    return Y_bouts_interp


def plot_bouts(Y_bouts_interp):
    colors = ["#008080", "#FF6F61", "#FFD700"]
    colors += [
        "#{:06x}".format(random.randint(0, 0xFFFFFF))
        for _ in range(300)
    ]
    print(len(colors))
    t_steps_interp = Y_bouts_interp.shape[2]
    interp_t = np.linspace(0, 1, t_steps_interp)

    plt.figure()
    for bout_idx, bout_data in enumerate(Y_bouts_interp):
        darkness_factor = 0.3 + 0.7 * (bout_idx / len(Y_bouts_interp))

        for i, color in enumerate(colors):
            plt.plot(interp_t, bout_data[i, :], color=color, marker='o', alpha=darkness_factor)
    plt.show()
