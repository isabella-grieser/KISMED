from models.basemodel import BaseModel
import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os

class ExampleModel(BaseModel):
    def __init__(self):
        self.th_opt = 0

    def train(self, train_data, train_label, val_data, val_label, fs):
        detectors = Detectors(fs)  # Initialisierung des QRS-Detektors
        sdnn_normal = np.array([])  # Initialisierung der Feature-Arrays
        sdnn_afib = np.array([])
        for idx, ecg_lead in enumerate(train_data):
            r_peaks = detectors.hamilton_detector(ecg_lead)  # Detektion der QRS-Komplexe
            sdnn = np.std(np.diff(
                r_peaks) / fs * 1000)  # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden
            if train_label[idx] == 'N':
                sdnn_normal = np.append(sdnn_normal, sdnn)  # Zuordnung zu "Normal"
            if train_label[idx] == 'A':
                sdnn_afib = np.append(sdnn_afib, sdnn)  # Zuordnung zu "Vorhofflimmern"
            if (idx % 100) == 0:
                print(str(idx) + "\t EKG Signale wurden verarbeitet.")
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].hist(sdnn_normal, 2000)
        axs[0].set_xlim([0, 300])
        axs[0].set_title("Normal")
        axs[0].set_xlabel("SDNN (ms)")
        axs[0].set_ylabel("Anzahl")
        axs[1].hist(sdnn_afib, 300)
        axs[1].set_xlim([0, 300])
        axs[1].set_title("Vorhofflimmern")
        axs[1].set_xlabel("SDNN (ms)")
        axs[1].set_ylabel("Anzahl")
        plt.show()

        sdnn_total = np.append(sdnn_normal, sdnn_afib)  # Kombination der beiden SDNN-Listen
        p05 = np.nanpercentile(sdnn_total, 5)  # untere Schwelle
        p95 = np.nanpercentile(sdnn_total, 95)  # obere Schwelle
        thresholds = np.linspace(p05, p95, num=20)  # Liste aller möglichen Schwellwerte
        F1 = np.array([])
        for th in thresholds:
            TP = np.sum(sdnn_afib >= th)  # Richtig Positiv
            TN = np.sum(sdnn_normal < th)  # Richtig Negativ
            FP = np.sum(sdnn_normal >= th)  # Falsch Positiv
            FN = np.sum(sdnn_afib < th)  # Falsch Negativ
            F1 = np.append(F1, TP / (TP + 1 / 2 * (FP + FN)))  # Berechnung des F1-Scores

        th_opt = thresholds[np.argmax(F1)]  # Bestimmung des Schwellwertes mit dem höchsten F1-Score
        self.th_opt = th_opt

        if os.path.exists("model.npy"):
            os.remove("model.npy")
        with open('model.npy', 'wb') as f:
            np.save(f, th_opt)

        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].hist(sdnn_normal, 2000)
        axs[0].set_xlim([0, 300])
        tmp = axs[0].get_ylim()
        axs[0].plot([th_opt, th_opt], [0, 10000])
        axs[0].set_ylim(tmp)
        axs[0].set_title("Normal")
        axs[0].set_xlabel("SDNN (ms)")
        axs[0].set_ylabel("Anzahl")
        axs[1].hist(sdnn_afib, 300)
        axs[1].set_xlim([0, 300])
        tmp = axs[1].get_ylim()
        axs[1].plot([th_opt, th_opt], [0, 10000])
        axs[1].set_ylim(tmp)
        axs[1].set_title("Vorhofflimmern")
        axs[1].set_xlabel("SDNN (ms)")
        axs[1].set_ylabel("Anzahl")
        plt.show()

    def test(self, test_data, test_labels, fs):
        detectors = Detectors(fs)
        r_peaks = detectors.hamilton_detector(test_data)  # Detektion der QRS-Komplexe
        peaks_A = []
        peaks_N = []
        for d, l in zip(r_peaks, test_labels):
            if l == "N":
                peaks_N.append(d)
            if l == "A":
                peaks_A.append(d)

        sdnn_A = np.std(np.diff(peaks_A) / fs * 1000)
        sdnn_N = np.std(np.diff(peaks_N) / fs * 1000)

        TP = np.sum(sdnn_A >= self.th_opt)  # Richtig Positiv
        FP = np.sum(sdnn_N >= self.th_opt)  # Falsch Positiv
        FN = np.sum(sdnn_A < self.th_opt)  # Falsch Negativ


        return TP / (TP + 1 / 2 * (FP + FN))