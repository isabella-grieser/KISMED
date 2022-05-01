from ecgdetectors import Detectors

"""
use the functions in this file only after filtering the input signals
"""
def average_f_wave_fluctuation(signals):
    """calculate the average fibrillatory wave fluctuation"""


def rri(signals, fs):
    """calculate the maximum variation of length of a hearbeat"""
    detectors = Detectors(fs)
    rris = []
    for s in signals:
        peaks = detectors.hamilton_detector(s)
        rr = []
        for i in range(1, len(peaks)):
            rr.append(peaks[i] - peaks[i-1])
        if rr:
            min_peak = min(rr)
            max_peak = max(rr)
            rris.append(max_peak - min_peak)
        else:
            #TODO: no feature handling
            rris.append(-1)
    return rris

