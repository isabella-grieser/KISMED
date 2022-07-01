from scipy import signal

#from preprocessing.augmentation import add_noise_samples
from preprocessing.augmentation import add_noise_samples
from utils.utils import get_all_data
from preprocessing.padding import divide_signal
from preprocessing.preprocessing import preprocess
from models.freqcnnmodel import FreqCNNModel
from sklearn.metrics import classification_report
from config import *
from utils.crossvalidation import *


def bool_to_label(v):
    if v:
        return 1
    elif not v:
        return 0
    else:
        return 0

def train(train_data, train_labels, fs, crossval=True, train=True):

    train_labels = ['O' if t != '~' else t for t in train_labels]
    if crossval:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess(ecg_leads, ecg_labels,
                                                                                            typ=TYPE, val_data=False)
    else:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess(ecg_leads, ecg_labels,
                                                                                                typ=TYPE)

    signals, _ = divide_signal(train_data[0], train_labels[0], DATA_SIZE, LOWER_DATA_SIZE_LIMIT)
    _, _, sprectogram = signal.spectrogram(signals[0], fs=fs, nperseg=64, noverlap=32)

    freqmodel = FreqCNNModel(fs, sprectogram.shape, TYPE)

    if train:
        if crossval:
            freqmodel = crossvalidation(freqmodel, fs, train_data, train_labels, typ=TYPE,
                                        path="model_weights/freqcnn/crossval/")
            model_path = f"model_weights/freqcnn/noise/model-{MODEL_VERSION}.hdf5"
            freqmodel.model.save_weights(model_path)
        else:
            freqmodel.train(train_data, train_labels, val_data, val_labels, fs)


if __name__ == '__main__':
    ecg_leads, ecg_labels, fs, ecg_names = get_all_data()
    # ecg_leads, ecg_labels = add_noise_samples(ecg_leads, ecg_labels, fs=300, duration=30)

    train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess(ecg_leads, ecg_labels, typ=TYPE, val_data=False)

    signals, labels = divide_signal(train_data[0], train_labels[0], DATA_SIZE, LOWER_DATA_SIZE_LIMIT)
    _, _, sprectogram = signal.spectrogram(signals[0], fs=fs, nperseg=64, noverlap=32)
    model = FreqCNNModel(fs, sprectogram.shape, ProblemType.NOISE)

    noise_pred = [bool_to_label(boo) for boo in model.is_noise(test_data)]
    noise_true = [bool_to_label(y == '~') for y in test_labels]

    print(classification_report(noise_true, noise_pred))