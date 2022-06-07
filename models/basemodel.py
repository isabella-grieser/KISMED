
class BaseModel:
    """
        Abstract class for model training/evaluation
        all models should ideally inherit from this class
    """

    def __init__(self):
        pass

    def train(self, train_data, train_labels, val_data, val_labels, fs, typ):
        """
        Train the model with the given data
        Parameters
        ----------
        train_data :list[numpy.array]
                    training EKG data
        train_labels:list[str]
                    training EKG labels
        val_data :  list[numpy.array]
                    validation EKG data
        val_labels: list[str]
                    validation EKG labels
        fs :        int
                    sampling frequency
        """
        pass

    def test(self, test_data, test_labels, fs, typ):
        """
        Test the model with the given data

        Parameters
        ----------
        test_data : list[numpy.array]
                    test EKG data
        test_labels :list[str]
                    test EKG labels
        fs :        int
                    sampling frequency
        Returns
        -------
        score: dict
                F1 score of the test data
        """
        pass

    def predict(self, test_data, fs, typ):
        """
        Test the model with the given data

        Parameters
        ----------
        test_data : list[numpy.array]
                    test EKG data
        fs :        int
                    sampling frequency
        Returns
        -------
        score: list
        """
        pass
