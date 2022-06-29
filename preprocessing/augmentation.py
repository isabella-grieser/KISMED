from imblearn.over_sampling import SMOTE
import numpy as np
from config import *
from neurokit2.signal import signal_noise
from utils.utils import data_amount
import random


def smote_augmentation(signals, labels):
    """
    do smote augmentation to deal with data imbalance
    """
    # for the augmentation, it is necessary for all values that are fitted to have the same length
    oversample = SMOTE(random_state=SEED)
    data = np.array(signals)
    sigs, labls = oversample.fit_resample(data, labels)
    return sigs, labls

def over_sample(train_data, train_labels):
    # copied from sebastians cnn model
    ids = np.where(np.array(train_labels) == 'A')[0]
    choices = np.random.choice(ids, len(np.where(np.array(train_labels) == 'N')[0]))

    data_A_oversampled = np.array(train_data, dtype=object)[choices]
    labels_A_oversampled = np.array(train_labels)[choices]

    data_oversampled = np.concatenate(
        [data_A_oversampled, np.array(train_data, dtype=object)[np.where(np.array(train_labels) == 'N')[0]]],
        axis=0)
    labels_oversampled = np.concatenate(
        [labels_A_oversampled, np.array(train_labels)[np.where(np.array(train_labels) == 'N')[0]]], axis=0)

    order = np.arange(len(data_oversampled))
    np.random.shuffle(order)
    data_oversampled = data_oversampled[order]
    labels_oversampled = labels_oversampled[order]

    return (data_oversampled.tolist(), labels_oversampled.tolist())

def add_noise_samples(data, labels, fs=300, duration=30):
    """
    add noise data without other augmentation methods to improve "noisiness" of the noise data
    """
    amounts = data_amount(labels)

    max_amount = max(amounts)
    noise_amount = sum(1 for l in labels if l == '~')

    extra_noise_amount = max_amount - noise_amount

    if extra_noise_amount > 0:
        extra_noise = [signal_noise(duration=duration, sampling_rate=fs, beta=random.randint(-2, 2)) for i in range(extra_noise_amount)]
        extra_labels = ["~" for i in range(extra_noise_amount)]
        if type(data) == list:
            data.extend(extra_noise)
        else:
            data = np.append(data, extra_noise, axis=0)

        if type(labels) == list:
            labels.extend(extra_labels)
        else:
            labels = np.append(labels, extra_labels, axis=0)

    return data, labels
