import numpy as np

from models.basemodel import BaseModel
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from preprocessing.padding import *
from preprocessing.preprocessing import *
from utils.utils import *
from utils.plotutils import *

from config import *

"""
deep learning model based on Automated Atrial Fibrillation Detection using a Hybrid CNN-LSTM Network
on Imbalanced ECG Datasets by Petmezas et al.
"""


class LSTMModel(BaseModel):

    def __init__(self, input_size):

        self.model_path = "checkpoint/lstm/binary/model.hdf5"
        super(BaseModel, self).__init__()

        # model definition
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(input_size, 1)))
        self.model.add(
            keras.layers.Conv1D(input_size, kernel_size=3, strides=2, activation='relu'))  # convolution layer 1
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(
            keras.layers.Conv1D(input_size // 1.5, kernel_size=3, strides=2, activation='relu'))  # convolution layer 2
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(
            keras.layers.Conv1D(input_size // 2, kernel_size=3, strides=2, activation='relu'))  # convolution layer 3
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(keras.layers.LSTM(248, return_sequences=True, dropout=0.2))  # LSTM layer
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(128, activation='relu'))  # dense layer
        if TYPE == ProblemType.BINARY:
            self.model.add(keras.layers.Dense(2))
        else:
            self.model.add(keras.layers.Dense(4))
        self.model.add(keras.layers.Softmax())
        self.model.summary()

    def train(self, train_data, train_labels, val_data, val_labels, fs):
        # do the preprocessing
        plot_labels = train_labels
        train_data, train_labels = self.preprocess(train_data, train_labels, fs)
        val_data, val_labels = self.preprocess(val_data, val_labels, fs)

        plot_all_signals(train_data, plot_labels, title='train signals')

        # callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                self.model_path,
                save_weights_only=True
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_precision",
                mode='max',
                min_delta=0,
                patience=5,
                restore_best_weights=True,
            )

        ]
        # model compilation
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[
                tfa.metrics.F1Score(num_classes=2),
                keras.metrics.Accuracy(),
                keras.metrics.Precision(),
                keras.metrics.Recall()
            ]
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
        y_true = labels_to_encodings(test_labels)

        metrics = {
            "f1": f1_score(y_true=y_true, y_pred=y_pred),
            "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
            "precision": precision_score(y_true=y_true, y_pred=y_pred),
            "recall": recall_score(y_true=y_true, y_pred=y_pred)
        }
        return metrics

    def predict(self, test_data, fs):
        self.model.load_weights(self.model_path)
        y_pred = []
        for t in test_data:
            data, _ = self.preprocess([t], ["N"], fs)
            y = np.argmax(self.model.predict(data), axis=1)
            # majority voting
            values, counts = np.unique(y, return_counts=True)
            ind = np.argmax(counts)
            y_pred.append(values[ind])

        return y_pred

    def preprocess(self, data, labels, fs):
        # signals, labels = divide_all_signals(data, labels, DATA_SIZE)
        signals, labels = divide_all_signals_in_heartbeats(data, labels, fs)
        signals = [normalize_data(s) for s in signals]
        signals = np.stack(signals, axis=0)
        signal_len = len(signals)
        data_len = len(signals[0])
        signals = signals.reshape(signal_len, data_len)
        labels = keras.utils.to_categorical(labels_to_encodings(labels)).reshape(signal_len, -1)
        return signals, labels
