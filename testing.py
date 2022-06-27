from matplotlib import pyplot as plt

from preprocessing.labelchecker import return_mislabeled_data
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
    # signals, labels = signals[:70], labels[:70]

    bad_signals, bad_labels = return_mislabeled_data(signals, labels, fs=300, typ=ProblemType.FOUR_CLASS, model="freqcnn")

    plot_all_signals(bad_signals, bad_labels, title='wrongly labeled signals')

    # plot_all_signals(heartbeats, heartbeat_labels, title='heartbeats')



    #plot_signal(signals[15], plt)
    #plot_signal(remove_noise_butterworth(signals[15], fs), plt)
    #plt.show()
