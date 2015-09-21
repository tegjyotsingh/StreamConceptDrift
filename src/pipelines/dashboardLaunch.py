__author__ = 'tegjyot'

import pprint

from src.libs import global_constants
from src.pipelines import feedback_loop_pipeline

def DashboardLaunch():
    parameters={}
    parameters['model']={'model_type': 'SVM_LINEAR'}
    parameters['metric']='ACCURACY'

    parameters['ilds']={}
    parameters['ilds']={
        'filename': '../../data/em',
        'initial_train_size': 0.20,
        'chunk_size': 1,
        'slide_rate': 1
    }

    parameters['drift_detector']={}
    parameters['drift_detector']['method']= global_constants.DRIFT_DETECTION_METHODS[2]
    parameters['drift_detector']['parameters']=global_constants.DRIFT_PARAMETERS[parameters['drift_detector']['method']]['parameters']
    parameters['drift_detector']['retrain_examples']=250
    parameters['train_module']=global_constants.DRIFT_PARAMETERS[parameters['drift_detector']['method']]['retrain_module']

    parameters['evaluation']={}
    parameters['evaluation']['attributes_to_track']=['drift', 'performance', 'performance_new', 'performance_sofar'] # defaults to performance and drift
    parameters['evaluation']['display_metrics']=['drift', 'performance_sofar']# None if all attributes are to be tracked ['drift', 'performance_new']
    parameters['intermediate']=\
        {'print_counter': 1000}

    print 'Running Pipeline with parameters:'
    pp=pprint.PrettyPrinter()
    pp.pprint(parameters)

    intermediate_model=feedback_loop_pipeline.LaunchPipeline(parameters)


if __name__=='__main__':
    DashboardLaunch()