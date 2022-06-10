from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from config import *



def crossvalidation(model, fs, data, labels, times=10, typ):
    """
    does crossvalidation; calculates and returns the model with the best test results
    """

    kfold = StratifiedKFold(n_splits=times, shuffle=True, random_state=SEED)

    f1_score = []
    for index, train_index, test_index in enumerate(kfold.split(data, labels)):
        train_data, train_labels = data[train_index], labels[train_index]
        # divide the train data into train and val data
        val_data, test_data, val_labels, test_labels = train_test_split(data[test_index], labels[test_index],
                                                              train_size=0.5,
                                                              stratify=labels[test_index],
                                                              random_state=SEED,
                                                              shuffle=True)
        # change the model save path
        model.model_path = f"model_weights/lstm/crossval/{'binary' if typ == ProblemType.BINARY else 'multiclass'}/model-{MODEL_VERSION}-crossval{index}.hdf5"
        model.train(train_data, train_labels, test_data, test_labels, fs, typ)
        f1_score.append(model.test(test_data, test_labels, fs)["f1"])

    # get the model with the highest f1 score

    print()
