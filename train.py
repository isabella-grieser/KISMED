# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese

"""

import numpy as np
from models import *
from models.deepmlmodel import DeepMLModel
from models.examplemodel import ExampleModel
from wettbewerb import load_references

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads, ecg_labels, fs, ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

#TODO: preprocessing

#TODO: model training
model = ExampleModel()
model.train(ecg_leads, ecg_labels, [], [], fs, ecg_names)
#TODO: model testing

