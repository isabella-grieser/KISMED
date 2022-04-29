from models.basemodel import BaseModel


class DeepMLModel(BaseModel):
    def __init__(self):
        pass

    def train(self, train_data, train_labels, val_data, val_labels, fs):
        pass

    def test(self, test_data, test_labels, fs):
        pass
