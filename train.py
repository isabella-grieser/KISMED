from models.deepmlmodel import DeepMLModel
from wettbewerb import load_references
from preprocessing.preprocessing import *
from utils.utils import *
import time

if __name__ == '__main__':

    ecg_leads, ecg_labels, fs, ecg_names = load_references(folder="data/training") # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

    train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess(ecg_leads, ecg_labels)

    model = DeepMLModel()
    model.train(train_data, train_labels, val_data, val_labels, fs)

    #TODO: model testing
    start_time = time.time()
    model.test(train_data, train_labels, fs)
    pred_time = time.time() - start_time
    print(f'time needed for prediction calculation: {pred_time}')
