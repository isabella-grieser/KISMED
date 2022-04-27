
class BaseModel:
    """
        Abstract class for model training/evaluation
        all models should ideally inherit from this class
    """

    def __init__(self):
        pass

    def train(self, train_data, train_label, val_data, val_label, fs, ecg_names):
        pass

    def test(self, test):
        pass
