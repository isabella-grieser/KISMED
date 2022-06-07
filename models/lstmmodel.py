import numpy as np

from models.basemodel import BaseModel
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from preprocessing.padding import *
from preprocessing.preprocessing import *
from utils.utils import *
from utils.plotutils import *
from preprocessing.augmentation import *

from config import *

"""
deep learning model based on Automated Atrial Fibrillation Detection using a Hybrid CNN-biLSTM Network
"""


class LSTMModel(BaseModel):

    def __init__(self, input_size, typ=ProblemType.BINARY):

        super(BaseModel, self).__init__()

        self.model_path = f"model_weights/lstm/{'binary' if typ == ProblemType.BINARY else 'multiclass'}/model-{MODEL_VERSION}"
        self.num_classes = 2 if typ == ProblemType.BINARY else 4
        # model definition
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(input_size, 1)))
        if typ == ProblemType.FOUR_CLASS:
            self.model.add(
                keras.layers.Conv1D(input_size, kernel_size=6, strides=4, activation='relu'))  # convolution layer 1
        else:
            self.model.add(
                keras.layers.Conv1D(input_size, kernel_size=3, strides=2, activation='relu'))  # convolution layer 1
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.MaxPooling1D(pool_size=2))

        if typ == ProblemType.FOUR_CLASS:
            self.model.add(
                keras.layers.Conv1D(input_size // 1.5, kernel_size=6, strides=4, activation='relu'))  # convolution layer 2
        else:
            self.model.add(
                keras.layers.Conv1D(input_size // 1.5, kernel_size=3, strides=2, activation='relu'))  # convolution layer 2
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.MaxPooling1D(pool_size=2))

        if typ == ProblemType.FOUR_CLASS:
            self.model.add(
                keras.layers.Conv1D(input_size // 2, kernel_size=6, strides=4, activation='relu'))  # convolution layer 3
        else:
            self.model.add(
                keras.layers.Conv1D(input_size // 2, kernel_size=3, strides=2, activation='relu'))  # convolution layer 3
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.2)))  # LSTM layer
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(256, activation='relu'))  # dense layer
        self.model.add(keras.layers.Dropout(0.2))
        if typ == ProblemType.BINARY:
            self.model.add(keras.layers.Dense(2))
        else:
            self.model.add(keras.layers.Dense(4))
        self.model.add(keras.layers.Softmax())
        self.model.summary()

    def train(self, train_data, train_labels, val_data, val_labels, fs, typ=ProblemType.BINARY):
        # do the preprocessing
        plot_labels = train_labels
        train_data, train_labels = self.preprocess(train_data, train_labels, fs, train=True, typ=typ)
        val_data, val_labels = self.preprocess(val_data, val_labels, fs, train=False, typ=typ)

        plot_all_signals(train_data, plot_labels, title='train signals')

        # callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                self.model_path,
                save_weights_only=True,
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_recall",
                mode='max',
                min_delta=0,
                patience=15,
                restore_best_weights=True,
            )
        ]

        metrics = [
            tfa.metrics.F1Score(num_classes=self.num_classes),
            keras.metrics.Precision(),
            keras.metrics.Recall()
        ]
        if typ == ProblemType.BINARY:
            metrics.append(keras.metrics.BinaryAccuracy())
        else:
            metrics.append(keras.metrics.Accuracy())

        # model compilation
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )

        self.model.fit(train_data, train_labels,
                       batch_size=TRAIN_BATCH,
                       epochs=EPOCHS,
                       validation_data=(val_data, val_labels),
                       validation_batch_size=TEST_BATCH,
                       callbacks=callbacks
                       )

    def test(self, test_data, test_labels, fs, typ=ProblemType.BINARY):

        y_pred = labels_to_encodings(self.predict(test_data, fs, typ))
        y_true = labels_to_encodings(test_labels)

        average = "binary" if typ == ProblemType.BINARY else "weighted"
        metrics = {
            "f1": f1_score(y_true=y_true, y_pred=y_pred, average=average),
            "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
            "precision": precision_score(y_true=y_true, y_pred=y_pred, average=average),
            "recall": recall_score(y_true=y_true, y_pred=y_pred, average=average)
        }
        return metrics

    def predict(self, test_data, fs, typ=ProblemType.BINARY):
        self.model.load_weights(self.model_path)
        y_pred = []
        for t in test_data:
            data, _ = self.preprocess([t], ["N"], fs, train=False, type=typ)
            y = np.argmax(self.model.predict(data, verbose=0), axis=1)
            # majority voting
            values, counts = np.unique(y, return_counts=True)
            ind = np.argmax(counts)
            y_pred.append(values[ind])

        return encodings_to_labels(y_pred)

    def preprocess(self, signals, labels, fs, train=True, typ=ProblemType.BINARY):

        signals = [invert2(s) for s in signals]
        signals = [remove_noise_butterworth(s, fs) for s in signals]
        signals = [normalize_data(s) for s in signals]
        if typ == ProblemType.BINARY:
            signals, labels = divide_all_signals_in_heartbeats(signals, labels, fs)
        else:
            signals, labels = divide_all_signals_with_lower_limit(signals, labels, DATA_SIZE, LOWER_DATA_SIZE_LIMIT)

        signals = np.stack(signals, axis=0)
        signal_len = len(signals)
        data_len = len(signals[0])
        signals = signals.reshape(signal_len, data_len)
        if train:
            signals, labels = smote_augmentation(signals, labels)
        labels = keras.utils.to_categorical(labels_to_encodings(labels), num_classes=self.num_classes)
        return signals, labels
