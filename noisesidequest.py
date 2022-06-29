from scipy import signal

#from preprocessing.augmentation import add_noise_samples
from preprocessing.augmentation import add_noise_samples
from utils.utils import get_all_data
from preprocessing.padding import divide_signal
from preprocessing.preprocessing import preprocess
from models.freqcnnmodel import FreqCNNModel
from sklearn.metrics import classification_report
from config import *


def bool_to_label(v):
    if v:
        return 1
    elif not v:
        return 0
    else:
        return 0

if __name__ == '__main__':
    ecg_leads, ecg_labels, fs, ecg_names = get_all_data()
    # ecg_leads, ecg_labels = add_noise_samples(ecg_leads, ecg_labels, fs=300, duration=30)

    train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess(ecg_leads, ecg_labels, typ=TYPE, val_data=False)

    signals, labels = divide_signal(train_data[0], train_labels[0], DATA_SIZE, LOWER_DATA_SIZE_LIMIT)
    _, _, sprectogram = signal.spectrogram(signals[0], fs=fs, nperseg=64, noverlap=32)
    model = FreqCNNModel(fs, sprectogram.shape, ProblemType.FOUR_CLASS)

    noise_pred = [bool_to_label(boo) for boo in model.is_noise(test_data)]
    noise_true = [bool_to_label(y == '~') for y in test_labels]

    print(classification_report(noise_true, noise_pred))