__author__ = 'tegjyot'

from src.libs import data_infra

def retrain_module_ref_accuracy(X, Y, model_type, drift_detector=None, metric_tracked=None):
    model=data_infra.TrainModel(X,Y,model_type)
    reference_state=None
    if drift_detector is not None:
        reference_state=drift_detector.computeReferenceState(Y_actual=Y,Y_pred=data_infra.PredictModel(model,X), metric=metric_tracked)
    return model, {'ref_accuracy':reference_state}

def retrain_module_ddm(X, Y, model_type, drift_detector=None, metric_tracked=None):
    model=data_infra.TrainModel(X,Y,model_type)
    reference_state={'ref': None}
    return model, reference_state