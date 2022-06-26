
from models.basemodel import BaseModel
import numpy as np
from scipy.signal import find_peaks
#import seaborn as sns
import math
import neurokit2 as nk
import scipy.io
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from scipy import signal, stats
import pickle
#from wettbewerb import load_references
from preprocessing.preprocessing import *
from preprocessing.padding import *
from preprocessing.features import *
from utils.utils import *


class rfclassifier(BaseModel):
    
    def __init__(self, fs, typ = ProblemType.BINARY):
        
        super(BaseModel, self).__init__()
        
        self.model_path = "model_weights/randomforest/rf_model3.pkl"
        self.scaler_path = "model_weights/randomforest/scaler5.pkl"
        self.num_classes = 2
        self.typ = typ
        
        self.scaler = preprocessing.MinMaxScaler()
        self.model = RandomForestClassifier(n_estimators = 700, max_depth = 50, random_state=0)
        
    
    def train(self, train_data, train_labels, val_data, val_labels, fs, typ):
        train_data, train_labels = self.preprocess(train_data, train_labels, fs)
        val_data, val_labels = self.preprocess(val_data, val_labels, fs)
        train_data = scaler.fit_transform(train_data)
        model = model.fit(train_data, train_labels)
        
        
    def test(self, test_data, test_labels, fs, typ):
        pred = self.predict(test_data, fs=fs) #declare scaler
        
        average = "binary"
        
        metrics = {
            "f1": f1_score(y_true=test_labels, y_pred=pred, average=average),
            "accuracy": accuracy_score(y_true=test_labels, y_pred=pred),
            "precision": precision_score(y_true=test_labels, y_pred=pred, average=average),
            "recall": recall_score(y_true=test_labels, y_pred=pred, average=average)
        }
        return metrics
        
        
    def predict(self, data, fs):
        dummy_labels = np.zeros(len(data))
        model = pickle.load(open(self.model_path, 'rb'))   # load saved model
        scaler = pickle.load(open(self.scaler_path, 'rb'))
        sig, label = self.preprocess(data, dummy_labels, fs)
        data = scaler.transform(sig)
        pred = model.predict(data)
        
        return pred
           
            
    def preprocess(self, signals, labels, fs=300):
        X, y = [], []
        for i in range(len(signals)):
            signal = invert2(signals[i])
            signal = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit")
            #sig = remove_noise_butterworth(sig, 300)
            r_peaks, p_peaks = extract_r_p_peaks(signal, fs)
            len_r = len(r_peaks)
            len_p = len(p_peaks)
            amp_r = [signal[i] for i in r_peaks]
            amp_p = [signal[i] for i in p_peaks]
            sd1, sd2, ratio = poincare_sd(r_peaks)

            cumdiff_r = np.diff(r_peaks) 

            mean1, std1, q2_1, iqr1, quartile_coeff_disp1 = stats_features(cumdiff_r)
            mean2, std2, q2_2, iqr2, quartile_coeff_disp2 = stats_features(amp_r)
            mean3, std3, q2_3, iqr3, quartile_coeff_disp3 = stats_features(amp_p)

            score1, score2, score3 = distribution_score(r_peaks, signal)

            dom_freq, energy_percent_at_dom = dominant_freq(signal)
            beat_rate = beats_per_sec(signal, len_r)

            array = [(len_r - len_p), score1, score2, score3, sd1, sd2, ratio, beat_rate, dom_freq, energy_percent_at_dom, mean1, 
                     std1, q2_1, iqr1, quartile_coeff_disp1, mean2, std2, q2_2, iqr2, quartile_coeff_disp2, mean3, std3, q2_3, 
                     iqr3, quartile_coeff_disp3]

            X.append(array)
            y.append(labels[i])
            
        data, labels = np.array(X), np.array(y)

        return data, labels
        
        

