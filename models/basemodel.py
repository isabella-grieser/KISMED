
class BaseModel:
    """
        Abstract class for model training/evaluation
        all models should ideally inherit from this class
    """

    def __init__(self):
        pass

    def train(self, train_data, train_label, val_data, val_label, fs):
        """
        Train the model with the given data
        Parameters
        ----------
        train_data :list[numpy.array]
                    training EKG data
        train_label :list[str]
                    training EKG labels
        val_data :  list[numpy.array]
                    validation EKG data
        val_label : list[str]
                    validation EKG labels
        fs :        int
                    sampling frequency
        """
        pass

    def test(self, test_data, test_labels):
        """
        Test the model with the given data

        Parameters
        ----------
        test_data : list[numpy.array]
                    test EKG data
        test_label :list[str]
                    test EKG labels
        Returns
        -------
        score: float
                F1 score of the test data
        """
        pass
