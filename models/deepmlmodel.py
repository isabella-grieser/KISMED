from models.basemodel import BaseModel
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.metrics import f1_score, accuracy_score, recall_score
from preprocessing.padding import *

from config import *
"""
deep learning model based on Automated Atrial Fibrillation Detection using a Hybrid CNN-LSTM Network
on Imbalanced ECG Datasets by Petmezas et al.
"""
class DeepMLModel(BaseModel):

    def __init__(self, input_size):
        super(BaseModel, self).__init__()
        # model definition
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(input_size,)))
        self.model.add(keras.layers.Conv1D(input_size//2, 3, stride=2))                 # convolution layer 1
        self.model.add(keras.BatchNormalization())
        self.model.add(keras.layers.MaxPooling1D(pool_size=2, stride=2))

        self.model.add(keras.layers.Conv1D(input_size//4, input_size//4, 3, stride=2))  # convolution layer 2
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.MaxPooling1D(pool_size=2, stride=2))

        self.model.add(keras.layers.Conv1D(input_size//8, input_size//8, 3, stride=2))  # convolution layer 3
        self.model.add(keras.BatchNormalization())
        self.model.add(keras.layers.MaxPooling1D(pool_size=2, stride=2))

        self.model.add(keras.layers.LSTM(64, 2))                                        # LSTM layer
        self.model.add(keras.layers.Dense(128))                                         # dense layer
        self.model.add(keras.layers.Softmax())                                                 # softmax activation function


    def train(self, train_data, train_labels, val_data, val_labels, fs):
        #do the preprocessing
        train_data, train_labels = self.__preprocess(train_data, train_labels)
        val_data, val_labels = self.__preprocess(val_data, val_labels)

        #callbacks
        callbacks = []
        #model compilation
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                            loss=keras.losses.BinaryCrossentropy(),
                            metrics=[tfa.metrics.F1Score]
                           )

        self.model.fit(train_data, train_labels,
                       batch_size=TRAIN_BATCH,
                       epochs=EPOCHS,
                       validation_data=(val_data, val_labels),
                       validation_batch_size=TEST_BATCH,
                       callbacks=callbacks
                       )

    def test(self, test_data, test_labels, fs):
        y_pred = self.predict(test_data, fs)

        metrics = {
            "f1": f1_score(y_true=test_labels, y_pred=y_pred),
            "accuracy": accuracy_score(y_true=test_labels, y_pred=y_pred),
            "recall": recall_score(y_true=test_labels, y_pred=y_pred)
        }
        return metrics

    def predict(self, test_data, fs):
        return self.model.predict(test_data)

    def __preprocess(self, data, labels):
        #TODO: preprocess
        return data, labels

