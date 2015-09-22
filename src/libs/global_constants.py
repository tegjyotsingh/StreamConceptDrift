__author__ = 'tegjyot'

from src.libs import drift_detection_lib
from src.libs import retrain_modules


COLORS=['red','blue','green','orange','yellow']

DRIFT_DETECTION_METHODS=[drift_detection_lib.NoDrift, drift_detection_lib.SimpleAccuracyDriftDetector, drift_detection_lib.DriftDetectionMethod]

DRIFT_PARAMETERS={
    drift_detection_lib.NoDrift: { 'parameters': {
        'MIN_SAMPLES': 30
    },
    'retrain_module': retrain_modules.retrain_module_ref_accuracy
    },
    drift_detection_lib.SimpleAccuracyDriftDetector:{ 'parameters': {
        'DROP_IN_ACCURACY': 0.1,
        'MIN_SAMPLES': 200 # dependendt on chunk size
    },
    'retrain_module': retrain_modules.retrain_module_ref_accuracy
    },
    drift_detection_lib.DriftDetectionMethod:{ 'parameters': {
        'MIN_SAMPLES': 30
    },
    'retrain_module': retrain_modules.retrain_module_ddm}
}
