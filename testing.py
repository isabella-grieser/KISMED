from matplotlib import pyplot as plt

from preprocessing.preprocessing import remove_noise_iir
from utils.plotutils import *
from wettbewerb import load_references
from preprocessing.features import *

if __name__ == '__main__':

    signals, labels, fs, _ = load_references(folder="data/training")

    plot_all_signals(signals, labels)
    print([len(s) for s in signals])
    plot_clusters(rri_max(signals, fs), labels, "rri_max")
    plt.show()
    plot_clusters(rri_std(signals, fs), labels, "rri_std")
    plt.show()

