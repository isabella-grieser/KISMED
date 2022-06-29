import pickle

from config import *
from tensorflow import keras

from models.rfclassifier import RfClassifier
import numpy as np
from cleanlab.filter import find_label_issues


def return_mislabeled_data(data, labels, fs=300, typ=ProblemType.BINARY):
    """
    try to find mislabeled data

    only used for "N" and "A" signals since they are the most important signals

    Was not used since it could not be finished on time
    """
    data = np.array([d for d, l in zip(data, labels) if l == "N" or l == "A"])
    labels = np.array([l for l in labels if l == "N" or l == "A"])

    model = RfClassifier()
    rfModel = pickle.load(open(model.model_path, 'rb'))  # load saved model

    preprocessed_data, cat_labels = model.preprocess(data, labels, fs)

    probs = rfModel.predict_proba(preprocessed_data)
    print(cat_labels)
    print(probs)
    print(len(cat_labels))
    print(probs.shape)
    issues = find_label_issues(cat_labels, probs)

    return data[issues], labels[issues]
