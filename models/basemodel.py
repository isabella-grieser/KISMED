
class BaseModel:
    """
        Abstract class for model training/evaluation
        all models should ideally inherit from this class
    """

    def __init__(self):
        pass

    def train(self, train_data, train_label, val_data, val_label, fs, ecg_names):
        """
        Train the model with the given data
        Parameters
        ----------
        train_data :
                    training EKG data
        train_label :
                    training EKG labels
        val_data :
                    validation EKG data
        val_label :
                    validation EKG labels
        fs :
                    sampling frequency
        ecg_names :  str, optional
                    name
        """
        pass

    def test(self, test_data, test_labels):
        """
        Test the model with the given data

        Parameters
        ----------
        test_data :
                    test EKG data
        test_label :
                    test EKG labels
        Returns
        -------
        score: float
                F1 score of the test data
        """
        pass
