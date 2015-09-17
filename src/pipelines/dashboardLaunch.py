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
        'filename': '../../data/rh',
        'initial_train_size': 0.25,
        'chunk_size': 100,
        'slide_rate': 100
    }

    parameters['drift_detector']={}
    parameters['drift_detector']['method']= global_constants.DRIFT_DETECTION_METHODS[1]
    parameters['drift_detector']['parameters']={
        'DROP_IN_ACCURACY': 0.01,
        'MIN_SAMPLES': 1000
    }
    parameters['train_module']=feedback_loop_pipeline.retrain_module_ref_accuracy

    parameters['evaluation']=None # defaults to performance and drift
    parameters['intermediate']=\
        {'print_counter': 100}

    print 'Running Pipeline with parameters:'
    pp=pprint.PrettyPrinter()
    pp.pprint(parameters)

    feedback_loop_pipeline.LaunchPipeline(parameters)

if __name__=='__main__':
    DashboardLaunch()