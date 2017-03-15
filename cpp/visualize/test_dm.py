import pml
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import os

import utils


def plot_dm(exp_dir):
    obs = pml.loadTxt(os.path.join(exp_dir, 'obs.txt'))
    cps = pml.loadTxt(os.path.join(exp_dir, 'cps.txt'))

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

    # Plot Observations
    ax = plt.subplot(gs[0])
    ax.set_title("Observations")
    utils.plot_dm(ax, obs, cps, 'b')
    ax.set_xticks([])

    # Plot Estimations
    dirs = ('filtering', 'smoothing', 'online_smoothing')
    for i, dirname in enumerate(dirs):
        mean = pml.loadTxt(os.path.join(exp_dir, dirname, 'mean.txt'))
        cpp = pml.loadTxt(os.path.join(exp_dir, dirname, 'cpp.txt'))
        ax = plt.subplot(gs[i+1])
        utils.plot_dm(ax, mean, cpp)
        ax.set_xticks([])
        ax.set_title(dirname)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    experiment_dir = '/tmp'
    if len(sys.argv) > 1:
        experiment_dir = sys.argv[1]
    plot_dm(experiment_dir)
