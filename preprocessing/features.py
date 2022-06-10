import numpy as np
from ecgdetectors import Detectors
from scipy import signal
import math
import neurokit2 as nk

"""
use the functions in this file only after filtering the input signals
"""
def average_f_wave_fluctuation(signals):
    """calculate the average fibrillatory wave fluctuation"""


def calculate_rri(signal, detector):
    peaks = detector.hamilton_detector(signal)
    rri = []
    for i in range(1, len(peaks)):
        rri.append(peaks[i] - peaks[i - 1])
    return rri

def rri_max(signals, fs):
    """calculate the maximum variation of length of a hearbeat"""
    detector = Detectors(fs)
    rris = [calculate_rri(s, detector) for s in signals]
    rri_max = []
    for rr in rris:
        if rr:
            min_peak = min(rr)
            max_peak = max(rr)
            rri_max.append(max_peak - min_peak)
        else:
            #TODO: no feature handling
            rri_max.append(-1)
    return rri_max

def rri_std(signals, fs):
    detector = Detectors(fs)
    rris = [calculate_rri(s, detector) for s in signals]
    rri_std = []
    for rr in rris:
        if rr:
            rri_std.append(np.std(rr))
        else:
            rri_std.append(-1)
    return rri_std

def find_peaks_r_p(sig, fs):
    _, rpeaks = nk.ecg_peaks(sig, sampling_rate=fs)
    rpeaks['ECG_R_Peaks'] = rpeaks['ECG_R_Peaks'][0:-1]  #Avoid indexing overflow error, Need to find robust method later
    _, waves_peak = nk.ecg_delineate(sig, rpeaks, sampling_rate=fs, method="dwt")
    r_peaks = rpeaks['ECG_R_Peaks']
    p_peaks = waves_peak['ECG_P_Peaks']
    p_peaks = [x for x in p_peaks if math.isnan(x) == False]
    return r_peaks, p_peaks  #return sample points of R and P peaks

def distribution_score(r_peaks, sig):
    first_peak = r_peaks[0]
    last_peak = r_peaks[-1]
    sig = sig[first_peak: last_peak+1]
    samples_per_beat = len(sig)/ len(r_peaks - 1)
    #Set threshold, Hardcoded for now, Need to automate later
    lower_thresh = samples_per_beat - 30    
    higher_thresh = samples_per_beat + 30
    cumdiff = np.diff(r_peaks) 
    #Number of R-R durations lie inside threshold
    len_center_idx = len([i for i in cumdiff if i > lower_thresh and i < higher_thresh])
    score =  len_center_idx/ len(cumdiff)
    return score

def poincare_sd(r_peaks):   #input argument r_peaks in samples only
    r_peaks = r_peaks*10/3  #Convert r_peaks from samples to time of occurence in millisec 
    rr = np.diff(r_peaks)
    rr_n = rr[:-1]
    rr_n1 = rr[1:]  #shifted rr duration
    sd1 = np.sqrt(0.5) * np.std(rr_n1 - rr_n)
    sd2 = np.sqrt(0.5) * np.std(rr_n1 + rr_n)
    return sd1, sd2, sd1/sd2    #return sd1, sd2 and their ratio

def dominant_freq(sig):
    f, Pxx_den = signal.periodogram(sig, 300)
    max_y = max(Pxx_den)    # Find the maximum y value
    max_x = f[Pxx_den.argmax()]  # Find the x value corresponding to the maximum y value
    return max_x  #returns dominant frequency in Hertz

def beats_per_sec(sig, num_rpeaks):
    duration_sec = len(sig)*(1/300)  #Duration of signal in seconds
    #beats = len(r_peaks)
    beats_per_sec = num_rpeaks/ duration_sec
    return beats_per_sec   #Generally Normal Beats_per_sec: 1 to 1.67, (Not always followed)

