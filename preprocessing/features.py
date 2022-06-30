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



def poincare_sd(r_peaks):  # input argument r_peaks in samples only
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
        mean = points.mean()
        std = points.std()
        q2 = np.quantile(points, 0.5)
        q1 = np.quantile(points, 0.25)
        q3 = np.quantile(points, 0.75)
        iqr = q3 - q1
        quartile_coeff_disp = (q3 - q1) / (q3 + q1)

    except Exception:
        mean, std, q2, iqr, quartile_coeff_disp = 0, 0, 0, 0, 0

    return mean, std, q2, iqr, quartile_coeff_disp


def dominant_freq(sig):
    try:
        f, Pxx_den = signal.periodogram(sig, 300)
        max_y = max(Pxx_den)  # Find the maximum y value
        max_x = f[Pxx_den.argmax()]  # Find the x value corresponding to the maximum y value
        energy_percentile = max_y / Pxx_den.sum()
    except Exception:
        max_x, energy_percentile = 0, 0
    return max_x, energy_percentile  # returns dominant frequency in Hertz


def beats_per_sec(sig, rpeaks):
    try:
        duration_sec = len(sig) * (1 / 300)  # Duration of signal in seconds
        # beats = len(r_peaks)
        beats_per_sec = rpeaks / duration_sec
    except Exception:
        beats_per_sec = 0
    return beats_per_sec  # Generally Normal Beats_per_sec: 1 to 1.67, (Not always followed)


def distribution_score(r_peaks, sig):
    try:
        first_peak = r_peaks[0]
        last_peak = r_peaks[-1]
        sig = sig[first_peak: last_peak + 1]
        samples_per_beat = len(sig) / (len(r_peaks) - 1)
        # Set threshold, Hardcoded for now, Need to automate later
        lower_thresh = samples_per_beat - 30
        higher_thresh = samples_per_beat + 30
        cumdiff = np.diff(r_peaks)
        # Number of R-R durations lie inside threshold
        len_center_idx = len([i for i in cumdiff if i > lower_thresh and i < higher_thresh])
        score1 = len_center_idx / len(cumdiff)
        ##################################################################################################
        max1 = np.quantile(cumdiff, 0.5)
        # max1 = x[np.argmax(y)]
        # print(max1)
        lower_thresh = max1 - 30
        higher_thresh = max1 + 30
        # Number of R-R durations lie inside threshold
        len_center_idx = len([i for i in cumdiff if i > lower_thresh and i < higher_thresh])
        score2 = len_center_idx / len(cumdiff)
        ##################################################################################################
        std = np.std(cumdiff)
        lower_thresh = max1 - std
        higher_thresh = max1 + std
        # Number of R-R durations lie inside threshold
        len_center_idx = len([i for i in cumdiff if i > lower_thresh and i < higher_thresh])
        score3 = len_center_idx / len(cumdiff)
        ###################################################################################################
    except Exception:
        score1, score2, score3 = 0, 0, 0
    return score1, score2, score3


def compute_FFT(sig, fs=300):
    spectrum = fftpack.fft(sig)
    spectrum = 2 / len(sig) * np.abs(spectrum[:len(sig) // 2])
    freq = np.linspace(0.0, fs / 2.0, len(sig) // 2)
    return (freq, spectrum)


def compute_PSD(sig, fs=300):
    spectrum = np.fft.fft(sig)
    factor = 2.0 / (len(spectrum) * len(spectrum))
    power = factor * (spectrum.real ** 2 + spectrum.imag ** 2)
    freq = np.fft.fftfreq(len(sig), 1 / fs)
    freq = freq[0:int(len(sig) / 2)]
    power = power[0:int(len(sig) / 2)]
    return (freq, power)


def compute_Band_Power(sig, lb, ub, fs=300, psd=False):
    if psd:
        ecg_freq, ecg_spectrum = compute_PSD(signal, fs)
    else:
        ecg_freq, ecg_spectrum = compute_FFT(sig, fs)
    mask = np.logical_and((ecg_freq > lb), (ecg_freq <= ub))
    band_power = np.sum(ecg_spectrum[mask] ** 2) / np.sum(ecg_spectrum ** 2)
    return band_power


def compute_beat_fib_power(ecg_lead, fs=300, invert=False, psd=False):
    used_ecg_lead = ecg_lead
    if invert: used_ecg_lead = invert2(ecg_lead)  # invert if needed
    # Compute normalized power in 'fib' frequency band for every heart beat
    beats = divide_heartbeats(used_ecg_lead, fs=fs)
    all_fib_powers = [compute_Band_Power(b, lb=3.8, ub=8) for b in beats]
    return all_fib_powers


def max_beat_fib_power(ecg, fs=300, invert=True, psd=False, calib_th=1):
    beat_fib_power = compute_beat_fib_power(ecg, fs, invert=invert, psd=psd)

    if len(beat_fib_power) == 0:
        return (-1)
    elif len(beat_fib_power) <= 2 * calib_th:
        return np.max(beat_fib_power)
    else:
        return np.max(beat_fib_power[calib_th:-calib_th])
