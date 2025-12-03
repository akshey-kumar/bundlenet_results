import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def behaviour_alignment(y_t, b_t, n_bins = 160, show_plot=False):
    # Binning the continuous behaviour

    bins = np.linspace(b_t[:,0].min(), b_t[:,0].max(), n_bins+1)
    b_inds = np.digitize(b_t[:,0], bins)
    y_b = {}
    y_b_std = {}

    i = 0
    # Iterating over discrete behaviour (direction)
    for discrete_b in [0,1]:
        # Iterating over continuous behaviour (position)
        for continuous_b in np.unique(b_inds):
            i = i+1
            idx_i = np.logical_and(
                b_t[:,1]==discrete_b,
                b_inds==continuous_b
            )
            if y_t[idx_i].size != 0:
                y_b[i] = y_t[idx_i].mean(axis=0)
                y_b_std[i] = y_t[idx_i].std(axis=0)

    y_b = pd.DataFrame({
        'b': [b for b in y_b],
        'y': [(y_b[b]) for b in y_b],
        'y_std': [(y_b_std[i]) for i in y_b_std],
    })

    if show_plot:
        plt.errorbar(
            np.stack(y_b['b'].to_numpy()),
            np.stack(y_b['y'].to_numpy())[:,1],
            yerr = np.stack(y_b['y_std'].to_numpy())[:,1]
                     )
        plt.show()
    return y_b
