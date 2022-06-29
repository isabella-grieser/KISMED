from sklearn.model_selection import GridSearchCV

from models.basemodel import BaseModel
import numpy as np
from scipy.signal import find_peaks
import math
import neurokit2 as nk
import scipy.io
import sklearn
import eli5 as eli
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from scipy import signal, stats
import pickle
from preprocessing.preprocessing import *
from preprocessing.padding import *
from preprocessing.features import *
from utils.utils import *
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="Mean of empty slice.")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")


class RfClassifier(BaseModel):

    def __init__(self):
        super(BaseModel, self).__init__()

        self.model_path = "model_weights/randomforest/rf_model.pkl"
        self.scaler_path = "model_weights/randomforest/scaler.pkl"

        self.scaler = preprocessing.MinMaxScaler()
        self.model = RandomForestClassifier(n_estimators=700, max_depth=50, random_state=SEED)

        self.feature_names = ['Number of p peaks missed', 'score1', 'score2', 'score3', 'sd1', 'sd2', 'ratio', 'beat_rate', 'dominant_freq',
                              'energy_percent_at_dominant_freq', 'mean1', 'std1', 'q2_1', 'iqr1', 'quartile_coeff_disp1', 'mean2', 'std2',
                              'q2_2', 'iqr2', 'quartile_coeff_disp2', 'mean3', 'std3', 'q2_3','iqr3', 'quartile_coeff_disp3']

        self.feature_description = {'Number of p peaks missed': 'Number of detected R peaks - Number of deteced P peaks', 
                                    'score1': 'Indicates proporion of R-R distances lies inside threshold (Range: 0 to 1, generally more value indicates N)',
                                    'score2': 'Indicates proporion of R-R distances lies inside threshold (Range: 0 to 1, generally more value indicates N)',
                                    'score3': 'Indicates proporion of R-R distances lies inside threshold (Range: 0 to 1, generally more value indicates N)',
                                    'sd1': 'short-term Heart rate variability' , 'sd2': 'long-term Heart rate variability', 'ratio': 'unpredictability of the RR' ,
                                    'beat_rate': 'Beats frequency based on R peaks', 'dominant_freq': 'None', 'energy_percent_at_dominant_freq': 'None', 'mean1': 'Mean of R-R distances',
                                    'std1': 'Standard deviation of R-R distances', 'q2_1': 'Second quarter of R-R distances',
                                    'iqr1': 'Inter quartile range of of R-R distances', 'quartile_coeff_disp1': 'Quartile cofficient dispersion of R-R distances',
                                    'mean2': 'Mean of R peaks amplitude', 'std2': 'Standard deviation of R peaks amplitude',
                                 'q2_2': 'Second quarter of R peaks amplitude', 'iqr2': 'Inter quartile range quarter of R peaks amplitude',
                                    'quartile_coeff_disp2': 'Quartile cofficient dispersion of R peaks amplitude', 'mean3': 'Mean of P peaks amplitude',
                                    'std3': 'Standard deviation of P peaks amplitude', 'q2_3': 'Second quartile of P peaks amplitude',
                                    'iqr3': 'Inter quartile range of P peaks amplitude' , 'quartile_coeff_disp3': 'Quartile cofficient dispersion of P peaks amplitude'


            }

        
    def train(self, train_data, train_labels, val_data, val_labels, fs, typ):
        train_data, train_labels = self.preprocess(train_data, train_labels, fs)
        val_data, val_labels = self.preprocess(val_data, val_labels, fs)

        train_data = self.scaler.fit_transform(train_data)
        self.model.fit(train_data, train_labels)

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

    def crossval(self, data, labels, fs, param_grid):

        grid_search = GridSearchCV(self.model, param_grid, cv=10, verbose=2, n_jobs=-1)

        ecg_leads, ecg_labels = self.preprocess(data, labels, fs)
        grid_search.fit(ecg_leads, ecg_labels)
        print(grid_search.best_estimator_)

    def explain_model(self):
        model = pickle.load(open(self.model_path, 'rb'))  # load saved model
        print(eli.explain_weights(model))

    def explain_prediction(self, signal):
        """
        explains the prediction for a single signal
        """
        model = pickle.load(open(self.model_path, 'rb'))  # load saved model
        print(eli.explain_prediction(model, signal))
