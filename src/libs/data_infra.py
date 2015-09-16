from sklearn import linear_model, svm, tree
import sklearn.metrics
import numpy as np

# reads from file and returns X, Y
def ReadFromFile(filename, type='csv'):
    with open(filename) as f:
        data = np.loadtxt(f, delimiter=",")
    X = np.array((data[:, 0:-1]))
    X.tolist()
    Y = np.array(data[:, -1])
    return X, Y


def SplitTrainAndTest(X, Y, train_ratio):
    train_size = int(len(Y) * train_ratio)
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]
    return [X_train, Y_train, X_test, Y_test]


def TrainModel(X, Y, model_type='SVM_LINEAR'):
    if model_type == 'SVM_LINEAR':
        model = svm.SVC(kernel='linear')
    else:
        raise Exception('Invalid Model')
    model.fit(X, Y)
    return model


def ComputePerf(Y_actual, Y_pred, metric = 'ACCURACY'):
    conf_matrix = sklearn.metrics.confusion_matrix(Y_actual, Y_pred)
    if metric == 'ACCURACY':
        metric_measure = sklearn.metrics.accuracy_score(Y_actual, Y_pred)
    return {'metric': metric, 'metric_measure': metric_measure, 'conf_matrix': conf_matrix}


def PredictModel(model, X_test):
    Y_pred = model.predict(X_test)
    return Y_pred


def TestPipeline():
    FILE = '../../data/rh'
    EVALUATION_METRIC = 'ACCURACY'
    TRAIN_RATIO = 0.25

    X, Y = ReadFromFile(FILE)
    [X_train, Y_train, X_test, Y_test] = SplitTrainAndTest( X, Y, TRAIN_RATIO)
    model = TrainModel(X_train, Y_train)

    performance = ComputePerf(PredictModel(model, X_train), Y_train, EVALUATION_METRIC)
    print 'Train Performance on %s: %s' % (performance['metric'],
                                           performance['metric_measure'])

    performance = ComputePerf(PredictModel(model, X_test), Y_test, EVALUATION_METRIC)
    print 'Test Performance on %s: %s' % (performance['metric'],
                                          performance['metric_measure'])

if __name__ == '__main__':
    TestPipeline()
