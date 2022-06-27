from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import signal
import neurokit2 as nk
from config import *


def preprocess(data, labels, typ=ProblemType.BINARY, val_data=True):
    if typ == ProblemType.BINARY:
        data = [d for d, l in zip(data, labels) if l == "N" or l == "A"]
        labels = [l for l in labels if l == "N" or l == "A"]
    else:
        pass
    if val_data:
        data_train, data_rest, y_train, y_rest = train_test_split(data, labels,
                                                                  train_size=TRAIN_SPLIT,
                                                                  stratify=labels,
                                                                  random_state=SEED,
                                                                  shuffle=True)
        data_val, data_test, y_val, y_test = train_test_split(data_rest, y_rest,
                                                              train_size=VAL_SPLIT / TRAIN_SPLIT,
                                                              stratify=y_rest,
                                                              random_state=SEED,
                                                              shuffle=True)
    else:
        data_train, data_test, y_train, y_test = train_test_split(data, labels,
                                                                  train_size=TRAIN_SPLIT,
                                                                  stratify=labels,
                                                                  random_state=SEED,
                                                                  shuffle=True)
        data_val, y_val = [], []
    return data_train, y_train, data_val, y_val, data_test, y_test


def remove_noise_butterworth(s, fs):
    # highpass filter to remove baseline wander noise
    highpass = signal.butter(7, Wn=0.5, btype="highpass", fs=fs, output="sos")
    first_filter_output = signal.sosfilt(highpass, s)
    # lowpass filter to remove high frequency noise
    lowpass = signal.butter(6, Wn=50, btype="lowpass", fs=fs, output="sos")
    output = signal.sosfilt(lowpass, first_filter_output)
    return output


def normalize_data(data):
    mini = np.min(data)
    maxi = np.max(data)
    return (data - mini) / (maxi - mini)


def normalize_data2(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def invert2(signal):
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=300)
    orig_sinal_peaks = signal[rpeaks['ECG_R_Peaks']]  # R-peaks of original signal

    inv_signal = 0 - signal
    _, rpeaks = nk.ecg_peaks(inv_signal, sampling_rate=300)
    inv_sinal_peaks = inv_signal[rpeaks['ECG_R_Peaks']]  # R-peaks of inverted signal

    # mean of R-peaks
    orig_peaks_mean = orig_sinal_peaks.mean()
    inv_peaks_mean = inv_sinal_peaks.mean()

    if orig_peaks_mean < inv_peaks_mean:
        out = inv_signal
    else:
        out = signal

    return out

