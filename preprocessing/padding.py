import numpy as np
from config import *
from ecgdetectors import Detectors
import neurokit2 as nk

minsize = 100


def zero_padding(signal, size, end_padding=False):
    padded_signal = np.array(signal)
    if len(signal) < size:
        if end_padding:
            padded_signal = np.pad(padded_signal, (0, (size - len(signal))), 'constant', constant_values=0)
        else:
            padded_signal = np.pad(padded_signal, ((size - len(signal)), 0), 'constant', constant_values=0)
    return padded_signal


def prev_val_padding(signal, size, end_padding=False):
    padded_signal = np.array(signal)
    if len(signal) < size:
        if end_padding:
            padded_signal = np.pad(padded_signal, (0, (size - len(signal))), 'constant', constant_values=signal[-1])
        else:
            padded_signal = np.pad(padded_signal, ((size - len(signal)), 0), 'constant', constant_values=signal[0])
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
        pass  # do nothing
    else:
        signals.append(zero_padding(signal, size))
    labels = [label for s in range(len(signals))]
    return signals, labels


def divide_all_signals(signals, labels, size):
    sigs = []
    labls = []
    for s, l in zip(signals, labels):
        sis, la = divide_signal(s, l, size)
        sigs.extend(sis)
        labls.extend(la)
    return sigs, labls


def divide_heartbeats(signal, fs):
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)

    start_size = int(fs * BF_PEAK_LEN * 10 ** (-3))
    end_size = int(fs * AFT_PEAK_LEN * 10 ** (-3))
    total_size = len(signal)

    heartbeats = []
    peaks = []

    if len(rpeaks['ECG_R_Peaks']) != 0:
        peaks = rpeaks['ECG_R_Peaks']
    else:
        # if the first package cannot find the peaks
        peaks = Detectors(fs).hamilton_detector(signal)

    # Fix: else 'list index out of range'
    if len(peaks) == 0:
        print('No R-Peaks found in signal!')
        # TODO: calling methods have to deal with []
        return []

    for p in peaks:
        start = p - start_size if p > start_size else 0
        end = p + end_size if p + end_size < total_size else total_size
        heartbeats.append(signal[start:end])

    # additional padding may be necessary
    beats = [prev_val_padding(h, start_size + end_size) for h in heartbeats[0:-1]]
    beats.append(prev_val_padding(heartbeats[-1], start_size + end_size, end_padding=True))

    return beats


def divide_all_signals_in_heartbeats(signals, labels, fs):
    sigs = []
    labls = []
    for s, l in zip(signals, labels):
        heartbeats = divide_heartbeats(s, fs)
        sigs.extend(heartbeats)
        labls.extend([l for i in range(len(heartbeats))])
    return sigs, labls
