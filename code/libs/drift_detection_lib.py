# Library for drift detection methdologies

import abc

import data_infra


class DriftDetector(object):
    __metaclass__ = abc.ABCMeta

    # Variables which can store necessary data
    State=None
    ReferenceState=None
    Parameters=None

    @abc.abstractmethod
    def Initialize(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def CheckForDrift(self, **kwargs):
        # data=None, predicted_labels=None, actual_labels=None
        raise NotImplementedError

class SimpleAccuracyDriftDetector(DriftDetector):
    def __init__(self, **kwargs):
        # Expects dictionaries of State and Parameters
        self.Initialize(**kwargs)

    def SetReferenceState(self, ref_accuracy=0.5):
        self.ReferenceState['ref_accuracy']=ref_accuracy

    def SetDefaultInitializeState(self, train_data= None):
        self.State['cycle_error']=0
        self.State['samples_processed']=0

    def SetParameters(self):
        self.Parameters['drop_in_accuracy']=0.25
        self.Parameters['min_samples']=10

    def Initialize(self, **kwargs):

        if 'State' not in kwargs:
            self.SetDefaultInitializeState()
        else:
            self.State=kwargs['State'].copy()
        if 'ReferenceState' not in kwargs:
            self.SetReferenceState()
        else:
            self.ReferenceState=kwargs['ReferenceState'].copy()
        if 'Parameters' not in kwargs:
            self.SetDefaultInitializeParameters()
        else:
            self.Parameters=kwargs['Parameters'].copy()

    def CheckForDrift(self, **kwargs):
        # predicted_labels, actual_labels
        # supervised so extracts the class labels
        for i, (label, actual_label) in enumerate(zip(kwargs['predicted_labels'],kwargs['actual_labels'])):
            if not (label == actual_label):
                self.State['cycle_error']+=1
            self.State['samples_processed']+=1
            if self.Parameters['min_samples'] > self.State['samples_processed']:
                continue
            else:
                if (self.State['cycle_error']/self.State['samples_processed'])-self.ReferenceState['ref_accuracy']>self.Parameters['drop_in_accuracy']:
                    # drift detected
                    return i+1
                else:
                    self.SetDefaultInitializeState()
        return -1

def TestSetup():
    FILE = '../data/rh'
    X, Y = ReadFromFile(FILE)
    [X_train, Y_train, X_test, Y_test] = data_infra.SplitTrainAndTest(data_infra.TRAIN_RATIO, X, Y)
    model = data_infra.TrainModel(X_train, Y_train)
    performance = data_infra.ComputePerf(data_infra.PredictModel(model, X_train), Y_train)
    ref_accuracy= performance['metric']
    return ref_accuracy, model, X_test, Y_test

def TestSimpleAccuracyDriftDetector():
    Parameters={'drop_in_accuracy': 0.25, 'min_samples': 10}
    ReferenceState={'ref_accuracy': 0.5}
    RetrainDrain=50
    reverse_count=RetrainDrain
    ref_accuracy, model, X_test, Y_test =TestSetup()
    sdd=SimpleAccuracyDriftDetector(Parameters=Parameters, ReferenceState=ReferenceState)

    for i,sample in enumerate(X_test):
        predicted_label= data_infra.PredictModel(model, sample)
        actual_label= Y_test[i]
        isdrift=sdd.CheckForDrift(predicted_label, actual_label)
        if isdrift != -1:
            # fix model
            if RetrainDrain>0:
                RetrainDrain-=1
            else:
                X_train=X_test[i-RetrianDrain,i]
                Y_train= Y_test[i-RetrainDrain,i]
                model= data_infra.TrainModel(X_train, Y_train)
                performance = data_infra.ComputePerf(data_infra.PredictModel(model, X_train), Y_train)
                ref_accuracy= performance['metric']
                sdd.SetReferenceState(ref_accuracy)




if __name__=='__main__':
    TestDriftDetectionMethodology()