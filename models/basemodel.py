
class BaseModel:
    """
        Abstract class for model training/evaluation
        all models should ideally inherit from this class
    """

    def __init__(self):
        pass

    def train(self, train, val):
        pass

    def test(self, test):
        pass
