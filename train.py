import sklearn

from models.freqcnnmodel import FreqCNNModel
from models.lstmmodel import LSTMModel
from models.rfclassifier import RfClassifier
from preprocessing.augmentation import add_noise_samples
from wettbewerb import load_references
from preprocessing.preprocessing import *
from preprocessing.padding import *
from utils.utils import *
from utils.crossvalidation import *
import time
from config import *


def train_freq_model(ecg_leads, ecg_labels, fs, crossval=True, train=True, to_evaluate=True):
    # train the FreqCNNModel
    if crossval:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess(ecg_leads, ecg_labels,
                                                                                            typ=TYPE, val_data=False)
    else:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess(ecg_leads, ecg_labels,
                                                                                                typ=TYPE)

    signals, labels = divide_signal(train_data[0], train_labels[0], DATA_SIZE, LOWER_DATA_SIZE_LIMIT)
    _, _, sprectogram = signal.spectrogram(signals[0], fs=fs, nperseg=64, noverlap=32)

    freqmodel = FreqCNNModel(fs, sprectogram.shape, TYPE)

    if train:
        if crossval:
            freqmodel = crossvalidation(freqmodel, fs, train_data, train_labels, typ=TYPE,
                                        path="model_weights/freqcnn/crossval/")
            model_path = f"model_weights/freqcnn/{'binary' if TYPE == ProblemType.BINARY else 'multiclass'}/model-{MODEL_VERSION}.hdf5"
            freqmodel.model.save_weights(model_path)
        else:
            freqmodel.train(train_data, train_labels, val_data, val_labels, fs)

    if to_evaluate:
        is_binary = TYPE == ProblemType
        evaluate(freqmodel, test_data, test_labels, is_binary=is_binary)


def train_rf_model(ecg_leads, ecg_labels, fs, crossval=True, train=True, to_evaluate=True):
    # train the Random Forest Classifier
    rfmodel = RfClassifier()

    train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess(ecg_leads, ecg_labels,
                                                                                        typ=ProblemType.BINARY,
                                                                                         val_data=False)
    if train:
        if crossval:
            param_grid = [
                {
                    'n_estimators': [i for i in range(100, 600, 50)],
                    'max_depth': [i for i in range(10, 70, 20)],
                    'max_features': ['sqrt', 'log2', 4, 8, None],
                    'min_samples_leaf': [1, 2, 4],
                    'min_samples_split': [5, 10],
                    'random_state': [SEED]
                }
            ]

            rfmodel.crossval(train_data, train_labels, fs, param_grid)
        else:
            rfmodel.train(train_data, train_labels, val_data, val_labels, fs, typ=ProblemType.BINARY)

    if to_evaluate:
        evaluate(rfmodel, test_data, test_labels, is_binary=True)


def evaluate(model, test_data, test_labels, is_binary=True):

    start_time = time.time()

    if is_binary:
        y_pred = model.predict(test_data, fs)

        y_pred = [1 if y == "N" else 0 for y in y_pred]
        test_data = [1 if y == "N" else 0 for y in test_labels]

        print(sklearn.metrics.classification_report(test_data, y_pred))

    y_pred = model.test(test_data, test_labels, fs, typ=ProblemType.FOUR_CLASS)
    print(y_pred)

    pred_time = time.time() - start_time
    print(f'time needed for prediction calculation: {pred_time}')


if __name__ == '__main__':

    # ecg_leads, ecg_labels, fs, ecg_names = load_references(folder="data/training") # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz),Name und Sampling-Frequenz 300 Hz
    # here we fetch all data that we gathered
    ecg_leads, ecg_labels, fs, ecg_names = get_all_data()
    # add additional noise samples
    ecg_leads, ecg_labels = add_noise_samples(ecg_leads, ecg_labels, fs=300, duration=30)

    # train_freq_model(ecg_leads, ecg_labels, fs, crossval=True, train=True, evaluate=True)
    train_rf_model(ecg_leads, ecg_labels, fs, crossval=False, train=False, to_evaluate=True)

