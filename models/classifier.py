from models.basemodel import BaseModel
from sklearn.svm import SVC
import pickle
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from config import *


class ClassifierModel(BaseModel):

    def __init__(self, *models, typ=ProblemType.BINARY):
        """
        initialize the classifier model
        models:   all models considered for the classifier; the models should already be trained
        """
        self.models = models
        self.model_path = f"model_weights/classifier/{'binary' if typ == ProblemType.BINARY else 'multiclass'}/model-{MODEL_VERSION}.hdf5"




    def train(self, train_data, train_labels, val_data, val_labels, fs, typ=ProblemType.BINARY):
        # get the output of all considered models

        y_pred = self.__calculate_input(train_data)
        y_pred.extend(self.__calculate_input(val_data))

        y_true = train_labels
        y_true.extend(val_data)
        classifier = SVC().fit(y_pred, y_true)

        with open(self.model_path, 'wb') as f:
            pickle.dump(classifier, f)

    def test(self, test_data, test_labels, fs, typ=ProblemType.BINARY):

        y_true = test_labels
        y_pred = self.predict(test_data, fs, typ)

        average = "binary" if typ == ProblemType.BINARY else "weighted"

        metrics = {
            "f1": f1_score(y_true=y_true, y_pred=y_pred, average=average),
            "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
            "precision": precision_score(y_true=y_true, y_pred=y_pred, average=average),
            "recall": recall_score(y_true=y_true, y_pred=y_pred, average=average)
        }
        return metrics

    def predict(self, test_data, fs, typ=ProblemType.BINARY):

        y_pred = self.__calculate_input(test_data)

        with open(self.model_path, 'rb') as f:
            classifier = pickle.load(f)

        return classifier.predict(y_pred)

    def __calculate_input(self, data):
        y_pred = []
        for d in data:
            y_temp = []
            for m in self.models:
                y_temp.append(m.predict(d))
            y_pred.append(y_temp)
        return y_pred