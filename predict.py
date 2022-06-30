# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from models.freqcnnmodel import FreqCNNModel
from config import *
from scipy import signal

from models.rfclassifier import RfClassifier
from preprocessing.padding import divide_signal
from utils.utils import *

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

#------------------------------------------------------------------------------
# Euer Code ab hier


    if is_binary_classifier:
        model = RfClassifier()

        y_pred = model.predict(ecg_leads, fs)
    else:
        signals, labels = divide_signal(ecg_leads[0], ["N"], DATA_SIZE, LOWER_DATA_SIZE_LIMIT)
        _, _, sprectogram = signal.spectrogram(signals[0], fs=fs, nperseg=64, noverlap=32)

        model = FreqCNNModel(fs, sprectogram.shape, ProblemType.FOUR_CLASS)

        y_pred = model.predict(ecg_leads, fs)

    predictions = list(zip(ecg_names, y_pred))
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
