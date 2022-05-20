from scipy.signal import lfilter
from sklearn.model_selection import train_test_split

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
    data_train, data_rest, y_train, y_rest = train_test_split(data, labels, train_size=TRAIN_SPLIT, stratify=True)
    data_val, data_test, y_val, y_test = train_test_split(data_rest, y_rest, train_size=VAL_SPLIT/TRAIN_SPLIT, stratify=True)
    return data_train, y_train, data_val, y_val, data_test, y_test

def remove_noise_iir(signal, n=15, b=1):
    # the larger n is, the smoother curve will be
    a = [1.0 / n] * n
    b = 1
    return lfilter(a, b, signal)


