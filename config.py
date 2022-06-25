from enum import Enum

class ProblemType(Enum):
    FOUR_CLASS = 1
    BINARY = 2

#data params
TRAIN_SPLIT = .9
VAL_SPLIT = .05
TEST_SPLIT = .05

DATA_SIZE = 9000
LOWER_DATA_SIZE_LIMIT = 500

SEED = 17


#model params
EPOCHS = 100
LEARNING_RATE = 1e-5
TRAIN_BATCH = 16
TEST_BATCH = 32
TYPE = ProblemType.FOUR_CLASS
MODEL_VERSION = "v1"

#heartbeat params (given a sampling frequency in Hz)
#in ms
BF_PEAK_LEN = 500
AFT_PEAK_LEN = 250

