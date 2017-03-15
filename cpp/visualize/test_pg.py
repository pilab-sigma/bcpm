import pml
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import os

import utils

def plot_pg(exp_dir):
    obs = pml.loadTxt(os.path.join(exp_dir, 'obs.txt'))
    cps = pml.loadTxt(os.path.join(exp_dir, 'cps.txt'))


    mean_filtering = pml.loadTxt(os.path.join(exp_dir,'filtering/mean.txt'))
    cpp_filtering = pml.loadTxt(os.path.join(exp_dir,'filtering/cpp.txt'))

    mean_smoothing = pml.loadTxt(os.path.join(exp_dir,'smoothing/mean.txt'))
    cpp_smoothing = pml.loadTxt(os.path.join(exp_dir,'smoothing/cpp.txt'))

    mean_online_smoothing = pml.loadTxt(os.path.join(exp_dir,
                                                   'online_smoothing/mean.txt'))
    cpp_online_smoothing = pml.loadTxt(os.path.join(exp_dir,
                                                    'online_smoothing/cpp.txt'))




    ax1.set_title("CPP Filtering")
    ax2.set_title("CPP Smoothing")
    ax3.set_title("CPP Online Smoothing")
    ax0.legend(['Hidden States', 'Observations', 'Filtering',
                'Smoothing', 'Online Smoothing'])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    experiment_dir = '/tmp'
    if len(sys.argv) > 1:
        experiment_dir = sys.argv[1]
    plot_pg(experiment_dir)




    states = states[0, :]
    obs = obs[0, :]
    mean = mean[0, :]
    mean2 = mean2[0, :]
    mean3 = mean3[0, :]
    T = len(states)

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(4, 1, height_ratios=[5, 1, 1, 1])

    ax0 = plt.subplot(gs[0])
    ax0.plot(range(T), obs, 'b-')
    ax0.plot(range(T), states, 'r-')
    ax0.plot(range(T), mean, 'g-')
    ax0.plot(range(T), mean2, 'm-')
    ax0.plot(range(T), mean3, 'c-')
    ax0.set_title("Sequence")
    ax0.set_xticks([])

    ax1 = plt.subplot(gs[1])
    ax1.bar(range(T), cpp)
    ax1.set_ylim([0, 1])
    ax1.set_xticks([])

    ax2 = plt.subplot(gs[2])
    ax2.bar(range(T), cpp2)
    ax2.set_ylim([0, 1])
    ax2.set_xticks([])

    ax3 = plt.subplot(gs[3])
    ax3.bar(range(T), cpp3)
    ax3.set_ylim([0, 1])

    if em_result == 'True' or em_result == '1' or em_result == True:
        ax1.set_title("CPP Smoothing - true parameters")
        ax2.set_title("CPP Smoothing - EM initial")
        ax3.set_title("CPP Smoothing - EM final")
        ax0.legend(['Hidden States', 'Observations', 'Mean Smoothing(True)',
                    'Mean Smoothing(EM initial)', 'Mean Smoothing(EM final)'])
    else:


    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_pg(sys.argv[1])