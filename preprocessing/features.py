import numpy as np
from ecgdetectors import Detectors
from scipy import signal, fftpack
import math
import neurokit2 as nk
from preprocessing.preprocessing import invert2
from preprocessing.padding import divide_heartbeats

"""
use the functions in this file only after filtering the input signals
"""

def extract_r_p_peaks(sig, fs):
    try:
        _, rpeaks = nk.ecg_peaks(sig, sampling_rate=fs)
        
        if len(rpeaks['ECG_R_Peaks']) > 4:
            rpeaks['ECG_R_Peaks'] = rpeaks['ECG_R_Peaks'][
                                    0:-1]  # Avoid indexing overflow error, Need to find robust method later
            _, waves_peak = nk.ecg_delineate(sig, rpeaks, sampling_rate=fs, method="cwt")
            r_peaks = rpeaks['ECG_R_Peaks']
            p_peaks = waves_peak['ECG_P_Peaks']
            if len(p_peaks) != 0:
                p_peaks = [x for x in p_peaks if math.isnan(x) is False]  
            return r_peaks, p_peaks  # return sample points of R and P peaks
        else:
            return [], []   
    except:
        return [], []



def poincare_sd(r_peaks):
    #input argument r_peaks in sample format only
    try:
        r_peaks = r_peaks * 10 / 3  # Convert r_peaks from samples to time of occurence in millisec
        rr = np.diff(r_peaks)
        rr_n = rr[:-1]
        rr_n1 = rr[1:]  # shifted rr duration
        sd1 = np.sqrt(0.5) * np.std(rr_n1 - rr_n)
        sd2 = np.sqrt(0.5) * np.std(rr_n1 + rr_n)
        if sd2 != 0:
            ratio = sd1 / sd2
        else:
            ratio = 1    
    except Exception:
        sd1, sd2, ratio = 0, 0, 1
    return sd1, sd2, ratio  # return sd1, sd2 and their ratio


def stats_features(points):
    try:
        points = np.array(points)
        mean = points.mean()    #Mean of points
        std = points.std()      # Standard deviation of points  
        q2 = np.quantile(points, 0.5)   #Second quartile of distribution of points
        q1 = np.quantile(points, 0.25)  #First quartile of distribution of points
        q3 = np.quantile(points, 0.75)  #Third quartile of distribution of points
        iqr = q3 - q1                   #Inter quartile range of distribution of points
        quartile_coeff_disp = (q3 - q1) / (q3 + q1) #Quartile coefficient dispersion of distribution of points

    except Exception:
        mean, std, q2, iqr, quartile_coeff_disp = 0, 0, 0, 0, 0

    return mean, std, q2, iqr, quartile_coeff_disp


def dominant_freq(sig):
    try:
        f, Pxx_den = signal.periodogram(sig, 300)
        max_y = max(Pxx_den)  # Find the maximum power
        max_x = f[Pxx_den.argmax()]  # Find the frequency corresponding to the maximum power
        energy_percentile = max_y / Pxx_den.sum()
    except Exception:
        max_x, energy_percentile = 0, 0
    return max_x, energy_percentile  # returns dominant frequency in Hertz and percenatge of energy at that frequency


def beats_per_sec(sig, rpeaks):
    try:
        duration_sec = len(sig) * (1 / 300)  # Duration of signal in seconds
        beats_per_sec = rpeaks / duration_sec
    except Exception:
        beats_per_sec = 0
    return beats_per_sec  # Generally Normal Beats_per_sec: 1 to 1.67


def distribution_score(r_peaks, sig):
    try:
        first_peak = r_peaks[0]
        last_peak = r_peaks[-1]
        sig = sig[first_peak: last_peak + 1]
        samples_per_beat = len(sig) / (len(r_peaks) - 1)

        ## calculate score 1 ##
        lower_thresh = samples_per_beat - 30
        higher_thresh = samples_per_beat + 30
        cumdiff = np.diff(r_peaks)
        len_center_idx = len([i for i in cumdiff if i > lower_thresh and i < higher_thresh])
        score1 = len_center_idx / len(cumdiff)
        # calculate score 2 ##
        max1 = np.quantile(cumdiff, 0.5)
        lower_thresh = max1 - 30
        higher_thresh = max1 + 30
        len_center_idx = len([i for i in cumdiff if i > lower_thresh and i < higher_thresh])
        score2 = len_center_idx / len(cumdiff)
        ## calculate score 3 ##
        std = np.std(cumdiff)
        lower_thresh = max1 - std
        higher_thresh = max1 + std
        len_center_idx = len([i for i in cumdiff if i > lower_thresh and i < higher_thresh])
        score3 = len_center_idx / len(cumdiff)
        
    except Exception:
        score1, score2, score3 = 0, 0, 0
    return score1, score2, score3


