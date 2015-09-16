import numpy as np
import pprint

from src.libs import data_infra
from src.libs import drift_detection_lib
from src.libs import oracle as truoracle
from src.libs import data_reader

# Pipeline for basic drift detection and handling
def Pipeline(parameters):
    # TODO: modify to class

    ##  Initialize Components
    #   Stream
    ilds=data_reader.InitiallyLabeledDataStream(parameters['stream']['filename'],
                                                  parameters['stream']['initial_train_size'],
                                                  parameters['stream']['chunk_size'],
                                                  parameters['stream']['slide_rate'])

    # Initial Model
    model=data_infra.TrainModel(ilds.initial_labeled_set['X'],ilds.initial_labeled_set['Y'],parameters['model']['model_type'])

    # Oracle
    oracle=truoracle.Oracle(ilds.parent_dataset.test_set['Y'])

    # Drift Detection
    # TODO make easy for other drift detectors to be substiturted and compared
    drift_detector=drift_detection_lib.SimpleAccuracyDriftDetector(parameters=parameters['drift_detector']['parameters'])
    # drift_detector_nodrift=drift_detection_lib.NoDrift()

    drift_detector.setReferenceState(ref_accuracy=data_infra.ComputePerf(ilds.initial_labeled_set['Y'],
                                                                         data_infra.PredictModel(
                                                                             model,ilds.initial_labeled_set['X']),
                                                                         parameters['metric'])['metric_measure'])

    ## Run Pipeline
    # TODO: evalaution class
    evaluation_measure=[]
    drifts_detected=[]

    while not ilds.stream.is_stream_end:
        X=ilds.stream.getUnlabaledData()
        if len(X) == 0: # last chunk might not receive data so dont compute anything
            break
        Y_pred=data_infra.PredictModel(model, X)
        Y_actual=ilds.stream.getEvaluationLabels()

        evaluation_measure.append(data_infra.ComputePerf(Y_actual, Y_pred, parameters['metric'])['metric_measure'])

        # feedback loop
        if drift_detector_nodrift.checkForDrift(predicted_labels=Y_pred, actual_labels=Y_actual) != -1:
            drifts_detected.append(1)
            # retrain -- Basic model retrain using half of current chunk
            high_range= ilds.stream.current_timestamp+parameters['stream']['chunk_size']
            low_range=int(high_range-(parameters['stream']['chunk_size']*0.5))
            Y_oracle= oracle.getTrueLabelRange(low_range, high_range)
            X_retrain=X[len(X)-(high_range-low_range):]
            model=data_infra.TrainModel(X_retrain,Y_oracle, parameters['model']['model_type'])
            drift_detector.setReferenceState(ref_accuracy=data_infra.ComputePerf(Y_oracle, data_infra.PredictModel(
                                                                             model,X_retrain),parameters['metric'])['metric_measure'])
        else:
            drifts_detected.append(0)


    ## Results
    # TODO :make class
    print evaluation_measure
    print drifts_detected
    print 'Sliding windows %d' %(len(evaluation_measure))
    print 'Mean performance on %s: %f' %(parameters['metric'],np.mean(evaluation_measure))
    print 'Number of drifts detected: %d' % sum(drifts_detected)
    print 'Expenditure by oracle: %d' % oracle.expenditure

def TestPipeline():

    parameters={}
    parameters['model']={'model_type': 'SVM_LINEAR'}
    parameters['metric']='ACCURACY'
    parameters['drift_detector']={}
    parameters['drift_detector']['parameters']={ 'DROP_IN_ACCURACY': 0.1,
                                                 'MIN_SAMPLES': 1000
                                                }
    parameters['stream']={}
    parameters['stream']={
        'filename': '../../data/rh',
        'initial_train_size': 0.25,
        'chunk_size': 1000,
        'slide_rate': 100
    }
    pp=pprint.PrettyPrinter()
    pp.pprint(parameters)

    Pipeline(parameters)

if __name__=='__main__':
    TestPipeline()
