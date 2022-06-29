import numpy as np

from models.basemodel import BaseModel
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from preprocessing.padding import *
from preprocessing.preprocessing import *
from utils.utils import *
from utils.plotutils import *
from preprocessing.augmentation import *
from scipy import signal

from config import *

"""
deep learning model based on ....
"""

CONV_BLOCK_AMOUNT = 7
FILTER_SIZE = 64
GROWTH = 32
class FreqCNNModel(BaseModel):

    def __init__(self, fs, dims, typ=ProblemType.BINARY):

        super(BaseModel, self).__init__()

        self.model_path = f"model_weights/freqcnn/{'binary' if typ == ProblemType.BINARY else 'multiclass'}/model-{MODEL_VERSION}.hdf5"
        self.num_classes = 2 if typ == ProblemType.BINARY else 4
        self.typ = typ
        self.dims = dims
        # model definition

        input_layer = keras.Input(shape=(dims[0], dims[1], 1))
        prev_layer = input_layer
        for i in range(CONV_BLOCK_AMOUNT):

            conv = keras.layers.Conv2D(filters=FILTER_SIZE + (i+1)*GROWTH, strides=(1, 1), kernel_size=(5, 5), padding='same')(prev_layer)
            batch = keras.layers.BatchNormalization()(conv)
            relu = keras.layers.ReLU()(batch)
            pool = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(relu)
            prev_layer = keras.layers.Dropout(0.1)(pool)

        # TODO: temporal average while considering zero padded data
        average = keras.layers.GlobalAveragePooling2D()(prev_layer)
        dense = None
        if typ == ProblemType.BINARY:
            dense = keras.layers.Dense(2)(average)
        else:
            dense = keras.layers.Dense(4)(average)

        output_layer = keras.layers.Softmax()(dense)

        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        self.model.summary()

    def train(self, train_data, train_labels, val_data, val_labels, fs, typ, version=""):
        # do the preprocessing
        train_data, train_labels = self.preprocess(train_data, train_labels, fs, train=True)
        val_data, val_labels = self.preprocess(val_data, val_labels, fs, train=False)

        # callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                self.model_path,
                save_weights_only=True,
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_precision"+version,
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
        if self.typ == ProblemType.BINARY:
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

    def test(self, test_data, test_labels, fs):

        y_pred = labels_to_encodings(self.predict(test_data, fs))
        y_true = labels_to_encodings(test_labels)

        average = "binary" if self.typ == ProblemType.BINARY else "weighted"
        metrics = {
            "f1": f1_score(y_true=y_true, y_pred=y_pred, average=average),
            "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
            "precision": precision_score(y_true=y_true, y_pred=y_pred, average=average),
            "recall": recall_score(y_true=y_true, y_pred=y_pred, average=average)
        }
        return metrics

    def predict(self, test_data, fs):
        self.model.load_weights(self.model_path)
        y_pred = []
        for t in test_data:
            data, _ = self.preprocess([t], ["N"], fs, train=False)
            y = np.argmax(self.model.predict(data, verbose=0), axis=1)
            # majority voting
            values, counts = np.unique(y, return_counts=True)
            ind = np.argmax(counts)
            y_pred.append(values[ind])

        return encodings_to_labels(y_pred)

    def preprocess(self, signals, labels, fs, train=True):

        signals = [remove_noise_butterworth(s, fs) for s in signals]
        signals, labels = divide_all_signals_with_lower_limit(signals, labels, DATA_SIZE, LOWER_DATA_SIZE_LIMIT)

        if train:
            signals, labels = smote_augmentation(signals, labels)

        # work with log spectogram
        spectograms = []
        for s in signals:
            _, _, spectogram = signal.spectrogram(s, fs=fs, nperseg=64, noverlap=32)
            spectogram = abs(spectogram)
            spectogram[spectogram > 0] = np.log(spectogram[spectogram > 0])
            spectogram = spectogram.reshape(-1, self.dims[0], self.dims[1], 1)
            spectograms.append(spectogram)
        labels = keras.utils.to_categorical(labels_to_encodings(labels), num_classes=self.num_classes)

        spectograms = np.array(spectograms).reshape(-1, self.dims[0], self.dims[1], 1)
        return spectograms, labels

    def is_noise(self, signals):
        """
        SIDEQUEST 2: find the noisy/noise signals

        returns a true/false list
        """
        y_pred = self.predict(signals, fs=300)
        return [y == '~' for y in y_pred]

