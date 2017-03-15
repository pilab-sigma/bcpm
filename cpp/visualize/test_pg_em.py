import pml
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys


def plot_pg(em_result=False):
    states = pml.loadTxt('/tmp/data/states.txt')
    obs = pml.loadTxt('/tmp/data/obs.txt')

    if em_result == 'True':
        mean = pml.loadTxt('/tmp/mean.txt')
        cpp = pml.loadTxt('/tmp/cpp.txt')

        mean2 = pml.loadTxt('/tmp/initial_mean.txt')
        cpp2 = pml.loadTxt('/tmp/initial_cpp.txt')

        mean3 = pml.loadTxt('/tmp/final_mean.txt')
        cpp3 = pml.loadTxt('/tmp/final_cpp.txt')
    else:



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
        ax1.set_title("CPP Filtering")
        ax2.set_title("CPP Smoothing")
        ax3.set_title("CPP Online Smoothing")
        ax0.legend(['Hidden States', 'Observations', 'Filtering',
                    'Smoothing', 'Online Smoothing'])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_pg(sys.argv[1])