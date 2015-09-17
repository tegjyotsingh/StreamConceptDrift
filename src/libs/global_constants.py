__author__ = 'tegjyot'

from src.libs import drift_detection_lib

COLORS=['red','blue','green','orange','yellow']
DRIFT_DETECTION_METHODS=[drift_detection_lib.NoDrift, drift_detection_lib.SimpleAccuracyDriftDetector]