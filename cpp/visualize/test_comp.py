import os
import sys

from bcpm_plots import *


def plot_coupled_model(dirname, M):

    states = load_txt(os.path.join(dirname, 'data/states.txt'))
    obs = load_txt(os.path.join(dirname, 'data/obs.txt'))

    mean1 = load_txt(os.path.join(dirname, 'filtering/mean.txt'))
    cpp1 = load_txt(os.path.join(dirname, 'filtering/cpp.txt'))

    mean2 = load_txt(os.path.join(dirname, 'smoothing/mean.txt'))
    cpp2 = load_txt(os.path.join(dirname, 'smoothing/cpp.txt'))

    mean3 = load_txt(os.path.join(dirname, 'online_smoothing/mean.txt'))
    cpp3 = load_txt(os.path.join(dirname, 'online_smoothing/cpp.txt'))

    pg_legend = ['Hidden States', 'Observations', 'Filtering',
                 'Smoothing', 'Online Smoothing']

    dm_titles = ['Filtering','Smoothing','Online Smoothing']

    cpp_titles = ['CPP Filtering', 'CPP Smoothing', 'CPP Online Smoothing']

    if M > 0:
        plot_dm(states[:M, :], obs[:M, :], mean1[:M, :], cpp1, mean2[:M, :],
                cpp2, mean3[:M, :], cpp3, dm_titles)
    else:
        plot_cpp(states, cpp1, cpp2, cpp3, cpp_titles)

    if M != states.shape[0]:
        plot_pg(states[M:, :], obs[M:, :], mean1[M:, :],
                mean2[M:, :], mean3[M:, :], pg_legend)


if __name__ == '__main__':
    plot_coupled_model(sys.argv[1], int(sys.argv[2]))
    plt.show()
