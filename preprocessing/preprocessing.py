from scipy.signal import lfilter


def preprocess(data, labels):
    """
    preprocess the data and divide it into train, val and test set
    Parameters
    ----------
    data : the total data
    labels : the total labels
    Returns
    ----------
    train data
    train labels
    val data
    val labels
    test data
    test labels
    -------

    """
    return data, labels, [], [], [], []

def remove_noise_iir(signal, n=15, b=1):
    # the larger n is, the smoother curve will be
    a = [1.0 / n] * n
    b = 1
    return lfilter(a, b, signal)
