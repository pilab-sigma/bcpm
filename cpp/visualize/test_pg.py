import os
import sys

from bcpm_plots import *


def plot_coupled_model(dirname):

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

    cpp_titles = ['CPP Filtering', 'CPP Smoothing', 'CPP Online Smoothing']

    plot_pg(states, obs, mean1, mean2, mean3, pg_legend)

    plot_cpp(states, cpp1, cpp2, cpp3, cpp_titles)

    plt.show()


if __name__ == '__main__':
    # By default, read fiels from /tmp
    dirname='/tmp'
    if len(sys.argv) == 2:
        dirname = sys.argv[1]

    plot_coupled_model(dirname)

