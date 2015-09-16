from sklearn import linear_model, svm, tree
import sklearn.metrics
import numpy as np

SVM_LINEAR_MODEL = 'SVM_LINEAR'
EVALUATION_TYPE = 'ACCURACY'
FILE = '../data/1CSurr.csv'
TRAIN_RATIO = 0.25


# reads from file and returns X, Y
def ReadFromFile(filename):
    with open(filename) as f:
        data = np.loadtxt(f, delimiter=",")
    X = np.array((data[:, 0:-1]))
    X.tolist()
    Y = np.array(data[:, -1])
    return X, Y


def SplitTrainAndTest(train_ratio, X, Y):
    train_size = int(len(Y) * train_ratio)
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]
    return [X_train, Y_train, X_test, Y_test]


def TrainModel(X, Y, model_type='SVM_LINEAR'):
    if model_type == SVM_LINEAR_MODEL:
        model = svm.SVC(kernel='linear')
    model.fit(X, Y)
    return model


def ComputePerf(Y_actual, Y_pred):
    conf_matrix = sklearn.metrics.confusion_matrix(Y_actual, Y_pred)
    if EVALUATION_TYPE == 'ACCURACY':
        metric = sklearn.metrics.accuracy_score(Y_actual, Y_pred)
    return {'metric': metric, 'conf_matrix': conf_matrix}


def PredictModel(model, X_test, Y_test=None):
    Y_pred = model.predict(X_test)
    return Y_pred

def TestPipeline():
    X, Y = ReadFromFile(FILE)
    [X_train, Y_train, X_test, Y_test] = SplitTrainAndTest(TRAIN_RATIO, X, Y)
    model = TrainModel(X_train, Y_train)

    performance = ComputePerf(PredictModel(model, X_train), Y_train)
    print 'Train Performance on %s: %s' % (EVALUATION_TYPE,
                                           performance['metric'])

    performance = ComputePerf(PredictModel(model, X_test), Y_test)
    print 'Test Performance on %s: %s' % (EVALUATION_TYPE,
                                          performance['metric'])

# Added for clustering portion
# ToDo

if __name__ == '__main__':
    TestPipeline()
