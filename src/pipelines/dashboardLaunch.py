__author__ = 'tegjyot'

import pprint
from inspect import getmembers
from sklearn.externals.six import StringIO
from sklearn import tree
import pydot
import src.libs.infra as infra


from src.libs import global_constants
from src.pipelines import feedback_loop_pipeline
from src.pipelines import continuous_build_pipeline


def DashboardLaunch():
    parameters={}
    parameters['model']={'model_type': 'SVM_LINEAR'} #SVM_LINEAR
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

    intermediate_models=feedback_loop_pipeline.LaunchPipeline(parameters)
    if parameters['model']['model_type']=='DT':
        for (model_ts,model) in intermediate_models.items():
            print 'Attributes for Tree %d' % model_ts
            with open("temp_results/"+str(model_ts)+".dot", 'w') as dotfile:
                tree.export_graphviz(model, dotfile)

    if parameters['model']['model_type']=='SVM_LINEAR':
        print 'Computing angles between the svm and previous one'
        for (model_ts,model) in sorted(intermediate_models.items()):
            print model_ts
            if model_ts==0:
                prev=list(model.intercept_)
                prev.extend(model.coef_[0])
                table_models=[]
                table_models.append(prev)
                continue
            current=list(model.intercept_)
            current.extend(model.coef_[0])
            for i,table_model in enumerate(table_models[-1:]):
                #print prev,current
                angle=infra.angle(table_model,current)
                print len(table_models), i, angle
            print '****'
            prev=current
            table_models.append(prev)




if __name__=='__main__':
    DashboardLaunch()