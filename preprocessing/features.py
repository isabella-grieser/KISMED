import numpy as np
from ecgdetectors import Detectors

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

