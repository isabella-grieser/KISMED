from imblearn.over_sampling import SMOTE
import numpy as np
from config import *


def smote_augmentation(signals, labels):
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