from sklearn.model_selection import GridSearchCV

from models.basemodel import BaseModel
import numpy as np
from scipy.signal import find_peaks
import math
import shap
import pandas as pd
import neurokit2 as nk
import scipy.io
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from scipy import signal, stats
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        self.explainer_path = "model_weights/randomforest/explainer.pkl"
        self.explanation_data_path = "model_weights/randomforest/explanation_data.csv"
        self.scaler = preprocessing.MinMaxScaler()
        self.model = RandomForestClassifier(max_depth=30, min_samples_split=5, n_estimators=350, random_state=SEED)

        self.explainer = None
        self.feature_names = ['Number of p peaks missed', 'score1', 'score2', 'score3', 'sd1', 'sd2', 'ratio', 'beat_rate', 'dominant_freq',
                              'energy_percent_at_dominant_freq', 'mean1', 'std1', 'q2_1', 'iqr1', 'quartile_coeff_disp1', 'mean2', 'std2',
                              'q2_2', 'iqr2', 'quartile_coeff_disp2', 'mean3', 'std3', 'q2_3', 'iqr3', 'quartile_coeff_disp3']

        self.feature_description = {'Number of p peaks missed': 'Number of detected R peaks - Number of deteced P peaks', 
                                    'score1': 'Indicates proporion of R-R distances lies inside threshold (Range: 0 to 1, generally more value indicates N)',
                                    'score2': 'Indicates proporion of R-R distances lies inside threshold (Range: 0 to 1, generally more value indicates N)',
                                    'score3': 'Indicates proporion of R-R distances lies inside threshold (Range: 0 to 1, generally more value indicates N)',
                                    'sd1': 'short-term Heart rate variability',
                                    'sd2': 'long-term Heart rate variability',
                                    'ratio': 'unpredictability of the RR',
                                    'beat_rate': 'Beats frequency based on R peaks',
                                    'dominant_freq': 'dominant frequency',
                                    'energy_percent_at_dominant_freq': 'TODO',
                                    'mean1': 'Mean of R-R distances',
                                    'std1': 'Standard deviation of R-R distances',
                                    'q2_1': 'Second quarter of R-R distances',
                                    'iqr1': 'Inter quartile range of of R-R distances',
                                    'quartile_coeff_disp1': 'Quartile cofficient dispersion of R-R distances',
                                    'mean2': 'Mean of R peaks amplitude',
                                    'std2': 'Standard deviation of R peaks amplitude',
                                    'q2_2': 'Second quarter of R peaks amplitude',
                                    'iqr2': 'Inter quartile range quarter of R peaks amplitude',
                                    'quartile_coeff_disp2': 'Quartile cofficient dispersion of R peaks amplitude',
                                    'mean3': 'Mean of P peaks amplitude',
                                    'std3': 'Standard deviation of P peaks amplitude',
                                    'q2_3': 'Second quartile of P peaks amplitude',
                                    'iqr3': 'Inter quartile range of P peaks amplitude',
                                    'quartile_coeff_disp3': 'Quartile cofficient dispersion of P peaks amplitude'
            }
        
    def train(self, train_data, train_labels, val_data, val_labels, fs, typ):

        train_data, train_labels = self.preprocess(train_data, train_labels, fs)

        self.check_data_means(train_data, train_labels)
        train_data = self.scaler.fit_transform(train_data)

        self.model.fit(train_data, train_labels)

        self.explainer = shap.TreeExplainer(self.model)

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(self.explainer_path, 'wb') as f:
            pickle.dump(self.explainer, f)

    def test(self, test_data, test_labels, fs):
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

        self.model = pickle.load(open(self.model_path, 'rb'))  # load saved model
        self.scaler = pickle.load(open(self.scaler_path, 'rb'))

        sig, _ = self.preprocess(data, dummy_labels, fs)
        data = self.scaler.transform(sig)

        pred = self.model.predict(data)

        return pred

    def preprocess(self, signals, labels, fs=300):
        X, y = [], []
        for i in range(len(signals)):
            # description of the features: see the feature description dictionary
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
        """
        Do crossvalidation for hyperparameter optimization
        """

        ecg_leads, ecg_labels = self.preprocess(data, labels, fs)

        grid_search = GridSearchCV(self.model, param_grid, cv=10, verbose=2, n_jobs=6)
        grid_search.fit(ecg_leads, ecg_labels)

        print(grid_search.best_estimator_)

    def check_data_means(self, train_data, labels):
        df = pd.DataFrame(train_data, columns=self.feature_names)
        df['label'] = labels

        normal_means = df[df['label'] == 'N'].median(axis = 0)
        atrial_means = df[df['label'] == 'A'].median(axis = 0)

        means = pd.concat([normal_means, atrial_means], axis=1).transpose()
        means['label'] =['N', 'A']
        means.to_csv(self.explanation_data_path, sep=';')


    def explain_prediction(self, signal, fs=300, show_feature_amount=5):
        """
        explains the prediction for a single signal
        """

        self.model = pickle.load(open(self.model_path, 'rb'))  # load saved model
        self.scaler = pickle.load(open(self.scaler_path, 'rb'))

        sig, _ = self.preprocess([signal], ["N"], fs)
        data = self.scaler.transform(sig)

        y_pred = self.model.predict(data)
        y_pred_probs = self.model.predict_proba(data)[0]
        
        self.explainer = pickle.load(open(self.explainer_path, 'rb'))  # load saved model
        

        print(f'label: {y_pred}')
        print(f'label probabilities: {self.model.classes_[0]}: {y_pred_probs[0]}, {self.model.classes_[1]}: {y_pred_probs[1]}')
        feature_vec, _ = self.preprocess([signal], ["N"])

        feature_vec = feature_vec[0]

        shap_values = self.explainer.shap_values(feature_vec)
        feature_importances = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(self.feature_names, feature_importances)),columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        most_important_feats = feature_importance['col_name'].to_numpy()[:show_feature_amount]

        importance = feature_importance['feature_importance_vals'].to_numpy()
        importance = importance / np.sum(importance) * 100

        # Create 2x2 sub plots
        plt.figure(0, figsize=(19,14))
        col = 16
        row = 24
        intro_space = plt.subplot2grid((row, col), (0, 0), colspan=col).axis('off')
        signal_plot = plt.subplot2grid((row, col), (1, 0), colspan=col//2, rowspan=row//2-1)
        freq_plot = plt.subplot2grid((row, col), (1,  col//2), colspan=col//2, rowspan=row//2-1)
        text_space = plt.subplot2grid((row, col), (row//2, 0), colspan=col, rowspan=row//2).axis('off')

        plt.suptitle("Explaination")
        plt.show()
        return most_important_feats, importance
