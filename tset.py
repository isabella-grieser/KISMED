from models.rfclassifier import RfClassifier
from preprocessing.preprocessing import *
from utils.utils import *
import time
from config import *


# ecg_leads, ecg_labels, fs, ecg_names = load_references(folder="data/training") # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz),Name und Sampling-Frequenz 300 Hz
ecg_leads, ecg_labels, fs, ecg_names = get_all_data()

train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess(ecg_leads, ecg_labels,
                                                                                    typ=ProblemType.BINARY,
                                                                                    val_data=False)
model = RfClassifier()
model.train(train_data, train_labels, val_data, val_labels, fs, typ=ProblemType.BINARY)

# TODO: model testing
start_time = time.time()
print(model.test(test_data, test_labels, fs, typ=ProblemType.BINARY))
pred_time = time.time() - start_time
print(f'time needed for prediction calculation: {pred_time}')

print(model.predict(test_data, fs))