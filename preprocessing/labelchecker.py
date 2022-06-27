from config import *
from tensorflow import keras

from preprocessing.padding import divide_signal, divide_all_signals_with_lower_limit, zero_padding
from preprocessing.preprocessing import remove_noise_butterworth
from utils.utils import labels_to_encodings
from models.freqcnnmodel import FreqCNNModel
from scipy import signal
import numpy as np
from cleanlab.filter import find_label_issues


def return_mislabeled_data(data, labels, fs=300, typ=ProblemType.BINARY, model="freqcnn"):
    preprocessed_data, cat_labels = preprocess(data, labels, fs)

    if model == "freqcnn":
        signals, labels = divide_signal(data[0], labels[0], DATA_SIZE, LOWER_DATA_SIZE_LIMIT)
        _, _, sprectogram = signal.spectrogram(signals[0], fs=fs, nperseg=64, noverlap=32)
        model = FreqCNNModel(fs, sprectogram.shape, typ)

        probs = model.model.predict(data, verbose=0)

    issues = find_label_issues(cat_labels, probs)

    return data[issues], labels[issues]

def preprocess(signals, labels, fs):
    """preprocessing function specially for the label checking"""
    signals = [remove_noise_butterworth(s, fs) for s in signals]
    max_len = max([len(s) for s in signals])
    signals = [zero_padding(s, max_len) for s in signals]

    # work with log spectogram
    spectograms = []
    for s in signals:
       _, _, spectogram = signal.spectrogram(s, fs=fs, nperseg=64, noverlap=32)
       spectogram = abs(spectogram)
       spectogram[spectogram > 0] = np.log(spectogram[spectogram > 0])
       spectograms.append(spectogram)

    spectograms = np.array(spectograms)
    return spectograms, np.array(labels_to_encodings(labels))