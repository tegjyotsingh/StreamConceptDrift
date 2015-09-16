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
    def initialize(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def checkForDrift(self, **kwargs):
        # data=None, predicted_labels=None, actual_labels=None
        raise NotImplementedError

    def printParameters(self):
        print 'Parameters of drift detection', self.Parameters

class NoDrift(DriftDetector):
    def __init__(self,**kwargs):
        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        pass

    def checkForDrift(self, **kwargs):
        return -1

class SimpleAccuracyDriftDetector(DriftDetector):
    def __init__(self, **kwargs):
        # Expects dictionaries of State, ReferenceState and Parameters
        self.initialize(**kwargs)

    def setReferenceState(self, ref_accuracy=0.5):
        self.ReferenceState={}
        self.ReferenceState['ref_accuracy']=ref_accuracy

    def setDefaultInitializeState(self, train_data= None):
        self.State={}
        self.State['cycle_error']=0
        self.State['samples_processed']=0

    def setDefaultParameters(self):
        self.Parameters={}
        self.Parameters['DROP_IN_ACCURACY']=0.25
        self.Parameters['MIN_SAMPLES']=10

    def initialize(self, **kwargs):

        if 'state' not in kwargs:
            self.setDefaultInitializeState()
        else:
            self.State=kwargs['state'].copy()
        if 'reference_state' not in kwargs:
            self.setReferenceState()
        else:
            self.ReferenceState=kwargs['reference_state'].copy()
        if 'parameters' not in kwargs:
            self.setDefaultParameters()
        else:
            self.Parameters=kwargs['parameters'].copy()

    def checkForDrift(self, **kwargs):
        # predicted_labels, actual_labels
        # supervised so extracts the class labels
        for i, (label, actual_label) in enumerate(zip(kwargs['predicted_labels'],kwargs['actual_labels'])):
            if not (label == actual_label):
                self.State['cycle_error']+=1
            self.State['samples_processed']+=1
            if self.Parameters['MIN_SAMPLES'] > self.State['samples_processed']:
                continue
            else:
                if self.ReferenceState['ref_accuracy']-\
                    (1-float(self.State['cycle_error'])/self.State['samples_processed'])>self.Parameters['DROP_IN_ACCURACY']:
                    # drift detected
                    self.setDefaultInitializeState()
                    return i+1 # position in chunk where drift detected
                else:
                    self.setDefaultInitializeState()
        return -1
