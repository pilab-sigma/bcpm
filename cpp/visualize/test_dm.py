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

    dm_titles = ['Filtering','Smoothing','Online Smoothing']

    plot_dm(states, obs, mean1, cpp1, mean2, cpp2, mean3, cpp3, dm_titles)

    plt.show()


if __name__ == '__main__':
    # By default, read fiels from /tmp
    dirname='/tmp'
    if len(sys.argv) == 2:
        dirname = sys.argv[1]

    plot_coupled_model(dirname)

