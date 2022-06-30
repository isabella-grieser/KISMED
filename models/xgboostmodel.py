
from models.basemodel import BaseModel
import numpy as np
from scipy.signal import find_peaks
import math
import neurokit2 as nk
import scipy.io
import sklearn
import eli5 as eli
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from scipy import signal, stats
import pickle
from preprocessing.preprocessing import *
from preprocessing.padding import *
from preprocessing.features import *
from utils.utils import *
import warnings
import xgboost as xgb
import pickle


warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="Mean of empty slice.")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")


class xgboostmodel(BaseModel):

    def __init__(self):
        super(BaseModel, self).__init__()

        self.model_path = "model_weights/xgboost/xgb2.pkl"
        self.scaler_path = "model_weights/xgboost/scaler5.pkl"
        self.encoder_path = "model_weights/xgboost/encoder2.pkl" 
        self.scaler = preprocessing.MinMaxScaler()


    def train(self, train_data, train_labels, val_data, val_labels, fs, typ):
        train_data, train_labels = self.preprocess(train_data, train_labels, fs)
        val_data, val_labels = self.preprocess(val_data, val_labels, fs)
        
        encoder = pickle.load(open(self.encoder_path, 'rb'))
        label_encoded_y = encoder.transform(train_labels)
        
        best_learning_rate = self.optimal_lr(train_data, label_encoded_y)
                
        train_data = self.scaler.fit_transform(train_data)
        model = xgb.XGBClassifier(objective="binary:logistic", learning_rate = best_learning_rate,
                                                   max_depth=10, n_estimators=100 ,random_state=SEED)
        model.fit(X, label_encoded_y)

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def test(self, test_data, test_labels, fs, typ):
        pred = self.predict(test_data, fs=fs)  # declare scaler
        average = "binary"

        metrics = {
            "f1": f1_score(y_true=test_labels, y_pred=pred, average=average, pos_label='A'),
            "accuracy": accuracy_score(y_true=test_labels, y_pred=pred),
            "precision": precision_score(y_true=test_labels, y_pred=pred, average=average, pos_label='A'),
            "recall": recall_score(y_true=test_labels, y_pred=pred, average=average, pos_label='A')
        }
        return metrics

    def predict(self, data, fs):
        dummy_labels = np.zeros(len(data))
        model = pickle.load(open(self.model_path, 'rb'))  # load saved model
        scaler = pickle.load(open(self.scaler_path, 'rb'))
        sig, label = self.preprocess(data, dummy_labels, fs)
        data = scaler.transform(sig)
        pred = model.predict(data)
        encoder = pickle.load(open(self.encoder_path, 'rb'))
        pred = encoder.inverse_transform(pred)

        return pred

    def preprocess(self, signals, labels, fs=300):
        X, y = [], []
        for i in range(len(signals)):
            signal = invert2(signals[i])
            signal = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit")
            # sig = remove_noise_butterworth(sig, 300)
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

            array = [(len_r - len_p), score1, score2, score3, sd1, sd2, ratio, beat_rate, dom_freq,
                     energy_percent_at_dom, mean1,
                     std1, q2_1, iqr1, quartile_coeff_disp1, mean2, std2, q2_2, iqr2, quartile_coeff_disp2, mean3, std3,
                     q2_3,
                     iqr3, quartile_coeff_disp3]

            X.append(array)
            y.append(labels[i])
            
        data, labels = np.array(X), np.array(y)
        data = data.astype('float64')
        data[np.isnan(data)] = 0
        
        return data, labels

    
# TO DO: Explainable AI for xgboost    
#     def explain_model(self):
#         model = pickle.load(open(self.model_path, 'rb'))  # load saved model
#         print(eli.explain_weights(model))
        
    def optimal_lr(self, X, y): 
        model = XGBClassifier(max_depth=10, n_estimators=100)
        learning_rate = [0.01, 0.1, 0.2, 0.3]
        param_grid = dict(learning_rate=learning_rate)

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
        grid_result = grid_search.fit(X, y)

        best_lr = grid_result.best_params_

        return best_lr['learning_rate']
