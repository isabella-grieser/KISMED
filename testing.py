from matplotlib import pyplot as plt

from utils.plotutils import *
from wettbewerb import load_references
from preprocessing.padding import *
from preprocessing.preprocessing import *
from preprocessing.features import *

if __name__ == '__main__':

    signals, labels, fs, _ = load_references(folder="data/training")
    signals = [d for d, l in zip(signals, labels) if l == "N" or l == "A"]
    labels = [l for l in labels if l == "N" or l == "A"]

    signals, labels = signals[5:10], labels[5:10]
    #plot_all_signals(signals, labels, title='normal signals')

    #signals = [zero_padding(s, size) for s in signals]
    #plot_all_signals(signals, labels, title='zero padded signals')
    #all_vals = [divide_signal(s, l, size//2) for s, l in zip(signals, labels)]
    #signals = [sd for s, l in all_vals for sd in s]
    #labels = [ld for s, l in all_vals for ld in l]
    #plot_all_signals(signals, labels, title='divided signals')
    signals = [invert2(d) for d in signals]
    signals = [normalize_data2(s) for s in signals]
    signals = [normalize_data(s) for s in signals]
    heartbeats, heartbeat_labels = divide_all_signals_in_heartbeats(signals, labels, fs)

    plot_all_signals(heartbeats, heartbeat_labels, title='heartbeats')

    #plot_signal(signals[15], plt)
    #plot_signal(remove_noise_butterworth(signals[15], fs), plt)
    #plt.show()
