# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese

"""

import numpy as np
from models import *
from models.deepmlmodel import DeepMLModel
from models.examplemodel import ExampleModel
from wettbewerb import load_references
from preprocessing import *

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads, ecg_labels, fs, ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

#TODO: preprocessing
train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess(ecg_leads, ecg_labels)

#TODO: model training
model = ExampleModel()
model.train(train_data, train_labels, val_data, val_labels, fs, ecg_names)

#TODO: model testing
model.test(test_data, test_labels)