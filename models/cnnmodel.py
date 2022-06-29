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
import pandas as pd  # TODO: can be later deleted

from config import *
import time  # for logger csv (SP)

"""
deep learning model based on Assessment of Potential Primary and Recurrent Ischemic Stroke by Detecting Atrial Fibrillation using
1D-CNN and CHA2DS2-VA Score by Mohammad Mahbubur Rahman Khan Mamun

DEPRECEATED; left in repository for documentation purposes
"""

# Some constants
TRAIN_RES_PATH = 'model_weights/cnn/binary/'
VERBOSE = 1
CNN_BATCH_SIZE = 128
CNN_NUM_EPOCHS = 3 #1000
MAX_SAMPLE_NUM = 18286

class CNNModel(BaseModel):

    def __init__(self, input_size):
        self.model_path = 'model_weights/cnn/binary/CNNModel_oversampled_1000epochs.h5'
        super(BaseModel, self).__init__()

        # model definition
        input_layer = keras.layers.Input(shape=(input_size, 1))

        conv1 = keras.layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(input_layer)
        maxp1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)

        conv2 = keras.layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(maxp1)
        maxp2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)

        conv3 = keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(maxp2)
        maxp3 = keras.layers.MaxPooling1D(pool_size=2)(conv3)

        conv4 = keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(maxp3)
        maxp4 = keras.layers.MaxPooling1D(pool_size=2)(conv4)

        conv5 = keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(maxp4)
        maxp5 = keras.layers.MaxPooling1D(pool_size=2)(conv5)

        conv6 = keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(maxp5)
        maxp6 = keras.layers.MaxPooling1D(pool_size=2)(conv6)
        drop1 = keras.layers.Dropout(0.5)(maxp6)

        conv7 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(drop1)
        maxp7 = keras.layers.MaxPooling1D(pool_size=2)(conv7)

        conv8 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(maxp7)
        maxp8 = keras.layers.MaxPooling1D(pool_size=2)(conv8)
        drop2 = keras.layers.Dropout(0.5)(maxp8)

        conv9 = keras.layers.Conv1D(filters=512, kernel_size=5, padding='same', activation='relu')(drop2)
        maxp9 = keras.layers.MaxPooling1D(pool_size=2)(conv9)
        drop3 = keras.layers.Dropout(0.5)(maxp9)

        conv10 = keras.layers.Conv1D(filters=512, kernel_size=5, padding='same', activation='relu')(drop3)

        flat1 = keras.layers.Flatten()(conv10)
        dense1 = keras.layers.Dense(64, activation='relu')(flat1)
        drop4 = keras.layers.Dropout(0.5)(dense1)
        dense2 = keras.layers.Dense(32, activation='relu')(drop4)

        output_layer = keras.layers.Dense(1, activation="sigmoid")(dense2)

        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        self.model.summary()

    def train(self, train_data, train_labels, val_data, val_labels, fs):
        # do the preprocessing
        plot_labels = train_labels
        train_data, train_labels = over_sample(train_data, train_labels)  # unbalanced set -> oversampling...
        train_data, train_labels = self.preprocess(train_data, train_labels, fs)
        val_data, val_labels = self.preprocess(val_data, val_labels, fs)

        #plot_all_signals(train_data, plot_labels, title='train signals')

        # callbacks
        #filename_weight = time.strftime("%Y%m%d-%H%M%S")+'-weights-improvement-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5'
        filename_weight = 'CNNModel_oversampled_1000epochs.h5'
        filename_log = time.strftime("%Y%m%d-%H%M%S")+'-performance-logs.csv'

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                TRAIN_RES_PATH+filename_weight,
                monitor='val_binary_accuracy',
                save_best_only=True,
                mode='max',
                save_freq='epoch',
                verbose=VERBOSE),

            keras.callbacks.EarlyStopping(
                monitor="val_precision",
                mode='max',
                min_delta=0,
                patience=5,
                restore_best_weights=True,
            ),

            keras.callbacks.CSVLogger(
                TRAIN_RES_PATH+filename_log,
                separator=',',
                append=False)
        ]
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00006),
            loss='binary_crossentropy',
            metrics=[
                tfa.metrics.F1Score(num_classes=1),
                keras.metrics.BinaryAccuracy(),
                keras.metrics.Precision(),
                keras.metrics.Recall()])

        history = self.model.fit(train_data, train_labels,
                        batch_size=CNN_BATCH_SIZE,
                        epochs=CNN_NUM_EPOCHS,
                        validation_data=(val_data, val_labels),
                        callbacks=callbacks)

        return history

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
            data, _ = self.preprocess([t], ['N'], fs)
            y_pred.append(np.rint(self.model.predict(data)[0]))

        return y_pred

    def preprocess(self, data, labels, fs):
        # preprocess data
        data_pad = [zero_padding(d, MAX_SAMPLE_NUM, end_padding=True) for d in data]
        data_pad = np.stack(data_pad, axis=0)
        signal_len = len(data_pad)
        data_len = len(data_pad[0])
        data_pad = data_pad.reshape(signal_len, data_len)
        data_pad = pd.DataFrame(data_pad).iloc[:, :MAX_SAMPLE_NUM-1]
        
        # preprocess labels
        labels = np.array(labels_to_encodings(labels)).reshape(signal_len, )

        return data_pad, labels
