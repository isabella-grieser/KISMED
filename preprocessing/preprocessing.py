from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
import numpy as np
import neurokit2 as nk
from config import *


def preprocess(data, labels):
    """
    preprocess the data and divide it into train, val and test set
    Parameters
    ----------
    data : the total data
    labels : the total labels
    Returns
    ----------
    train data
    train labels
    val data
    val labels
    test data
    test labels
    -------

    """
    if TYPE == ProblemType.BINARY:
        data = [d for d, l in zip(data, labels) if l == "N" or l == "A"]
        labels = [l for l in labels if l == "N" or l == "A"]

    data_train, data_rest, y_train, y_rest = train_test_split(data, labels,
                                                              train_size=TRAIN_SPLIT,
                                                              stratify=labels,
                                                              random_state=SEED)
    data_val, data_test, y_val, y_test = train_test_split(data_rest, y_rest,
                                                          train_size=VAL_SPLIT/TRAIN_SPLIT,
                                                          stratify=y_rest,
                                                          random_state=SEED)
    return data_train, y_train, data_val, y_val, data_test, y_test

def remove_noise_iir(signal, n=15, b=1):
    # the larger n is, the smoother curve will be
    a = [1.0 / n] * n
    return lfilter(a, b, signal)

def normalize_data(data):
    mini = np.min(data)
    maxi = np.max(data)
    return (data - mini) / (maxi - mini)

def invert2(signal):
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=300)
    orig_sinal_peaks = signal[rpeaks['ECG_R_Peaks']]     #R-peaks of original signal

    inv_signal = 0 - signal        
    _, rpeaks = nk.ecg_peaks(inv_signal, sampling_rate=300)                       
    inv_sinal_peaks = inv_signal[rpeaks['ECG_R_Peaks']]  #R-peaks of inverted signal

    #mean of R-peaks
    orig_peaks_mean = orig_sinal_peaks.mean()
    inv_peaks_mean = inv_sinal_peaks.mean()

    if orig_peaks_mean < inv_peaks_mean:
        out = inv_signal
    else:
        out = signal

    return out
