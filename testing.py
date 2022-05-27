from matplotlib import pyplot as plt

from utils.plotutils import *
from wettbewerb import load_references
from preprocessing.padding import *
from preprocessing.preprocessing import *
from preprocessing.features import *

if __name__ == '__main__':

    signals, labels, fs, _ = load_references(folder="data/training")

    #plot_all_signals(signals, labels, title='normal signals')

    #signals = [zero_padding(s, size) for s in signals]
    #plot_all_signals(signals, labels, title='zero padded signals')
    #all_vals = [divide_signal(s, l, size//2) for s, l in zip(signals, labels)]
    #signals = [sd for s, l in all_vals for sd in s]
    #labels = [ld for s, l in all_vals for ld in l]
    #plot_all_signals(signals, labels, title='divided signals')
    heartbeats = []
    heartbeat_labels = []
    for s, l in zip(signals, labels):
        h = divide_heartbeats(s, fs)
        heartbeats.extend(h)
        heartbeat_labels.extend([l for r in range(len(h))])

    # plot_all_signals(heartbeats[:100], labels[:100], title='heartbeats')

    plot_signal(signals[15], plt)
    plot_signal(remove_noise_butterworth(signals[15], fs), plt)
    plt.show()
