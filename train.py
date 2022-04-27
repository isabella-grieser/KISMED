from models.deepmlmodel import DeepMLModel
from models.examplemodel import ExampleModel
from wettbewerb import load_references
from preprocessing import *
from utils import *
import time

if __name__ == '__main__':

    ecg_leads, ecg_labels, fs, ecg_names = load_references(folder="data/training") # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

    #TODO: preprocessing
    train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess(ecg_leads, ecg_labels)

    calc_data_amount(ecg_labels)

    #TODO: model training
    model = DeepMLModel()
    model.train(train_data, train_labels, val_data, val_labels, fs)

    #TODO: model testing
    start_time = time.time()
    model.test(test_data, test_labels)
    pred_time = time.time() - start_time

    #preditions for the unlabeled data set
