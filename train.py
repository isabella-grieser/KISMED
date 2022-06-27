from models.freqcnnmodel import FreqCNNModel
from models.lstmmodel import LSTMModel
from preprocessing.augmentation import add_noise_samples
from wettbewerb import load_references
from preprocessing.preprocessing import *
from preprocessing.padding import *
from utils.utils import *
from utils.crossvalidation import *
import time
from config import *

CROSSVAL = True
if __name__ == '__main__':


    ecg_leads, ecg_labels, fs, ecg_names = load_references(folder="data/training") # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz),Name und Sampling-Frequenz 300 Hz
    # ecg_leads, ecg_labels, fs, ecg_names = get_all_data()
    # ecg_leads, ecg_labels = add_noise_samples(ecg_leads, ecg_labels, fs=300, duration=30)

    if CROSSVAL:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess(ecg_leads, ecg_labels, typ=TYPE, val_data=False)
    else:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess(ecg_leads, ecg_labels, typ=TYPE)

    signals, labels = divide_signal(train_data[0], train_labels[0], DATA_SIZE, LOWER_DATA_SIZE_LIMIT)
    _, _, sprectogram = signal.spectrogram(signals[0], fs=fs, nperseg=64, noverlap=32)
    model = FreqCNNModel(fs, sprectogram.shape, TYPE)
    if CROSSVAL:
        pass
        # model = crossvalidation(model, fs, train_data, train_labels, typ=TYPE, path="model_weights/freqcnn/crossval/")
    else:
        model.train(train_data, train_labels, val_data, val_labels, fs)

    #TODO: model testing
    start_time = time.time()
    print(model.test(test_data, test_labels, fs))
    pred_time = time.time() - start_time
    print(f'time needed for prediction calculation: {pred_time}')

    print(model.predict(test_data, fs))