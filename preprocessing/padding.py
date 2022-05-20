import numpy as np
from config import *
from ecgdetectors import Detectors
minsize = 100

def zero_padding(signal, size):
    padded_signal = np.array(signal)
    if len(signal) < size:
        padded_signal = np.pad(padded_signal, (0, (size - len(signal))), 'constant')
    return padded_signal

def divide_signal(signal, label, size):
    signals = []
    if len(signal) > size:
        signals.append(signal[:size])
        s, _ = divide_signal(signal[size:], label, size)
        signals.extend(s)
    elif len(signal) == size:
        signals.append(signal)
    elif len(signal) < minsize:
        pass # do nothing
    else:
        signals.append(zero_padding(signal, size))
    labels = [label for s in range(len(signals))]
    return signals, labels

def divide_heartbeats(signal, fs):
    peaks = Detectors(fs).hamilton_detector(signal)
    start_size = int(fs * BF_PEAK_LEN*10**(-3))
    end_size = int(fs * AFT_PEAK_LEN*10**(-3))
    total_size = len(signal)
    heartbeats = []
    for p in peaks:
        start = p - start_size if p > start_size else 0
        end = p + end_size if p + end_size < total_size else total_size
        heartbeats.append(signal[start:end])
    #for first and last heartbeat: additional padding may be necessary
    heartbeats[0] = zero_padding(heartbeats[0], start_size + end_size)
    heartbeats[len(heartbeats)-1] = zero_padding(heartbeats[len(heartbeats)-1], start_size + end_size)
    return heartbeats
