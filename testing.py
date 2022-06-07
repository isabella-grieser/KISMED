from matplotlib import pyplot as plt

from utils.plotutils import *
from utils.utils import *
from wettbewerb import load_references
from preprocessing.padding import *
from preprocessing.preprocessing import *
from preprocessing.features import *
from preprocessing.augmentation import *
from config import *
from tensorflow.python.client import device_lib
if __name__ == '__main__':

    signals, labels, fs, _ = load_references(folder="data/training")

    #signals = [d for d, l in zip(signals, labels) if l == "N" or l == "A"]
    #labels = [l for l in labels if l == "N" or l == "A"]
    print(device_lib.list_local_devices())  #
    signals, labels = signals[:70], labels[:70]
    signals, labels = divide_all_signals_with_lower_limit(signals, labels, DATA_SIZE, LOWER_DATA_SIZE_LIMIT)
    calc_data_amount(labels)
    signals, labels = smote_augmentation(signals, labels)

    plot_all_signals(signals, labels, title='normal signals')
    calc_data_amount(labels)

    #signals = [zero_padding(s, size) for s in signals]
    #plot_all_signals(signals, labels, title='zero padded signals')
    #all_vals = [divide_signal(s, l, size//2) for s, l in zip(signals, labels)]
    #signals = [sd for s, l in all_vals for sd in s]
    #labels = [ld for s, l in all_vals for ld in l]
    #plot_all_signals(signals, labels, title='divided signals')
    # signals = [invert2(d) for d in signals]
    # signals = [normalize_data2(s) for s in signals]
    # signals = [normalize_data(s) for s in signals]
    # heartbeats, heartbeat_labels = divide_all_signals_in_heartbeats(signals, labels, fs)

    # plot_all_signals(heartbeats, heartbeat_labels, title='heartbeats')



    #plot_signal(signals[15], plt)
    #plot_signal(remove_noise_butterworth(signals[15], fs), plt)
    #plt.show()
