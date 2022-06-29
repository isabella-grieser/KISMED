import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from config import *



def crossvalidation(model, fs, data, labels, times=10, typ=ProblemType.BINARY, path="model_weights/lstm/crossval/"):
    """
    does crossvalidation; calculates and returns the model with the best test results
    """

    if type(data) == list:
        data = np.array(data)
    if type(labels) == list:
        labels = np.array(labels)

    kfold = StratifiedKFold(n_splits=times, shuffle=True, random_state=SEED)
    index = 0
    f1_score = []
    for train_index, test_index in kfold.split(data, labels):
        train_data, train_labels = data[train_index], labels[train_index]
        test_data, test_labels = data[test_index], labels[test_index]

        # divide the train data into train and val data
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels,
                                                              train_size=0.5,
                                                              stratify=train_labels,
                                                              random_state=SEED,
                                                              shuffle=True)
        # change the model save path
        model.model_path = f"{path}{'binary' if typ == ProblemType.BINARY else 'multiclass'}/model-{MODEL_VERSION}-crossval{index}.hdf5"
        # this code part is used to get the early stopping to work :(
        version = "" if index == 0 else f"_{index}"

        model.train(train_data, train_labels, test_data, test_labels, fs, typ, version=version)
        f1_score.append(model.test(test_data, test_labels, fs)["f1"])

        index += 1

    # get the model with the highest f1 score
    print(f1_score)
    argmax = np.argmax(np.array(f1_score))
    print(f"best model: {argmax}")

    model.model_path = f"{path}{'binary' if typ == ProblemType.BINARY else 'multiclass'}/model-{MODEL_VERSION}-crossval{argmax}.hdf5"
    model.model.load_weights(model.model_path)
    return model
