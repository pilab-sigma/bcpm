import matplotlib.pyplot as plt
import numpy as np


def get_cps(states):
    cps = np.zeros((states.shape[1], 1))
    for i in range(1,len(cps)):
        if any(states[:, i] != states[:, i-1]):
            cps[i] = 1
    return cps


def plot_dm(ax, x, cpp, color='r'):
    [k ,t] = x.shape
    ax.pcolormesh(x, cmap=plt.cm.Greys, vmin=0, vmax=np.max(np.max(x)))
    ax.vlines(np.arange(0, t), 0, cpp*k,  colors=color,
              linestyles='-', linewidth=2, label='change point prob.')
    ax.set_yticks(np.arange(k) + 0.5)
    ax.set_yticklabels(np.arange(k) + 1)
