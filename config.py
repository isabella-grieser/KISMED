from enum import Enum

class ProblemType(Enum):
    FOUR_CLASS = 1
    BINARY = 2

#data params
TRAIN_SPLIT = .8
VAL_SPLIT = .1
TEST_SPLIT = .1

DATA_SIZE = 4500

SEED = 17


#model params
EPOCHS = 100
LEARNING_RATE = 1e-5
TRAIN_BATCH = 16
TEST_BATCH = 32
TYPE = ProblemType.BINARY
MODEL_VERSION = "v4"

#heartbeat params (given a sampling frequency in Hz)
#in ms
BF_PEAK_LEN = 500
AFT_PEAK_LEN = 250

