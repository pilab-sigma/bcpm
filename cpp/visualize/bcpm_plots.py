import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


def load_txt(filename):
    X = np.loadtxt(filename)
    dim = int(X[0])
    size = []
    for i in range(dim):
        size.append(int(X[i+1]))
    X = np.reshape(X[dim+1:], size, order='F')
    return X


def get_cps(states):
    cps = np.zeros((states.shape[1], 1))
    for i in range(1,len(cps)):
        if any(states[:, i] != states[:, i-1]):
            cps[i] = 1
    return cps


def plot_pg(states, obs, mean, mean2, mean3, legend):

    length = states.shape[1]
    num_components = states.shape[0]
    cps = get_cps(states)

    plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(num_components, 1,
                           height_ratios=np.ones(num_components))

    for n in range(num_components):
        ax = plt.subplot(gs[n])
        ax.set_title("Gamma Poisson Component " + str(n))
        y_lim_max = np.max(obs) * 1.1
        ax.plot(range(length), states[n, :], 'r-')
        ax.plot(range(length), obs[n, :], 'b-')
        ax.plot(range(length), mean[n, :], 'g-')
        ax.plot(range(length), mean2[n, :], 'm-')
        ax.plot(range(length), mean3[n, :], 'c-')
        ax.vlines(np.arange(0, length), 0, cps * y_lim_max,  colors='k',
                  linestyles='-', linewidth=2)

        ax.set_ylim([0, y_lim_max])
        if n != num_components-1:
            ax.set_xticks([])

        ax.legend(legend)


def plot_single_dm(ax, alpha, cpp, line_color='r'):
    K = alpha.shape[0]
    T = alpha.shape[1]
    ax.pcolormesh(alpha, cmap=plt.cm.Greys,
                  vmin=0, vmax=np.max(np.max(alpha)))
    ax.vlines(np.arange(0, T), 0, cpp*K,  colors=line_color, linestyles='-',
              linewidth=2, label='change point prob.')


def plot_dm(states, obs, mean, cpp, mean2, cpp2, mean3, cpp3, titles):
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

    ax = plt.subplot(gs[0])
    ax.set_title("Multinomial Observations")
    plot_single_dm(ax, obs, get_cps(states), 'b')
    ax.set_xticks([])

    ax = plt.subplot(gs[1])
    ax.set_title("Multinomial Mean - " + titles[0])
    plot_single_dm(ax, mean, cpp)
    ax.set_xticks([])

    ax = plt.subplot(gs[2])
    ax.set_title("Multinomial Mean - " + titles[1])
    plot_single_dm(ax, mean2, cpp2)
    ax.set_xticks([])

    ax = plt.subplot(gs[3])
    ax.set_title("Multinomial Mean - " + titles[2])
    plot_single_dm(ax, mean3, cpp3)


def plot_cpp(states, cpp1, cpp2, cpp3, titles):

    cps = get_cps(states)
    T = len(cps)
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

    ax0 = plt.subplot(gs[0])
    ax0.vlines(np.arange(0, T), 0, cps,  colors='k',
               linestyles='-', linewidth=2)
    ax0.set_ylim([0, 1])
    ax0.set_xticks([])
    ax0.set_title('Change points')

    ax1 = plt.subplot(gs[1])
    ax1.vlines(np.arange(0, T), 0, cpp1,  colors='r',
               linestyles='-', linewidth=2)
    ax1.set_ylim([0, 1])
    ax1.set_xticks([])
    ax1.set_title(titles[0])

    ax2 = plt.subplot(gs[2])
    ax2.vlines(np.arange(0, T), 0, cpp2,  colors='r',
               linestyles='-', linewidth=2)
    ax2.set_ylim([0, 1])
    ax2.set_xticks([])
    ax2.set_title(titles[1])

    ax3 = plt.subplot(gs[3])
    ax3.vlines(np.arange(0, T), 0, cpp3,  colors='r',
               linestyles='-', linewidth=2)
    ax3.set_ylim([0, 1])
    ax3.set_title(titles[2])
