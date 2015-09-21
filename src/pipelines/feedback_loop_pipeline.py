import numpy as np
import pprint

from src.libs import data_infra
from src.libs import drift_detection_lib
from src.libs import oracle as truoracle
from src.libs import data_reader
from src.libs import evaluation_engine as evaluate
from src.libs import global_constants
from src.libs import retrain_modules

class Pipeline(object):

    def __init__(self, **kwargs):
        # need an ilds, train module, oracle, drift detector, ealuation engine object
        self.ilds=kwargs['ilds']
        self.oracle=kwargs['oracle']
        self.drift_detector=kwargs['drift_detector']
        self.evaluation_engine=kwargs['evaluation_engine']
        self.train_module=kwargs['train_module']
        self.model=kwargs['model']
        self.parameters=kwargs['parameters']

        self.ReferenceState=None
        self.buffer_Y=[]
        self.buffer_X=[]
        self.IntermediateStoreModel={}

    def Start(self):

        intermediate_print_counter=self.parameters['intermediate']['print_counter']
        evaluated_samples=0
        print 'Starting Stream'
        while not self.ilds.stream.is_stream_end:

            X=self.ilds.stream.getUnlabaledData()

            number_samples=len(X)
            intermediate_print_counter-=number_samples
            evaluated_samples+=1
            #printing after every few samples
            if intermediate_print_counter<=0:
                print 'Evaluating %d' %(evaluated_samples)
                intermediate_print_counter=self.parameters['intermediate']['print_counter']

            if len(X) == 0: # last chunk might not receive data so dont compute anything
                break

            Y_pred=data_infra.PredictModel(self.model, X)
            Y_actual=self.ilds.stream.getEvaluationLabels()

            self.evaluation_engine.addRunningResults(performance=data_infra.ComputePerf(Y_actual,Y_pred
                                                                                            ,self.parameters['metric'])['metric_measure'])
            #evaluating only on newly occuring samples
            self.evaluation_engine.addRunningResults(performance_new=data_infra.ComputePerf(Y_actual[-self.parameters['ilds']['slide_rate']:]
                                                                                            ,Y_pred[-self.parameters['ilds']['slide_rate']:]
                                                                                            ,self.parameters['metric'])['metric_measure'])
            #computing averages so far
            n=len(self.evaluation_engine.running_performance['performance_sofar'])
            avg=self.evaluation_engine.running_performance['performance_sofar'][-1] if n!=0 else 0
            new_avg=(self.evaluation_engine.running_performance['performance'][-1]+n*avg)/float(n+1)
            self.evaluation_engine.addRunningResults(performance_sofar=new_avg)

            #TODO: add intermediate for last drift location

            #TODO: add intermediate for forgetting and last window
            flag_is_drift=False

            # feedback loop
            isDrift=self.drift_detector.checkForDrift(predicted_labels=Y_pred, actual_labels=Y_actual)
            if isDrift != -1:
                self.buffer_X.extend(X)
                self.buffer_Y.extend(Y_actual)
                if isDrift=='Drift':
                    # TODO implement better way of obtianing retrain samples

                    if len(self.buffer_Y)<self.parameters['drift_detector']['retrain_examples']:
                        self.evaluation_engine.addRunningResults(drift = 0)
                        continue

                    end=min(len(self.buffer_X), self.parameters['drift_detector']['retrain_examples'])
                    X_retrain=self.buffer_X[-1*end:]
                    Y_retrain=self.buffer_Y[-1*end:]
                    self.oracle.expenditure+=len(Y_retrain)


                    # need to buffer and or use active learning for streaming sequential data
                    # need to extend to work so that samples collected in case we need to add more training
                    #[X_retrain,  Y_oracle]=BufferRetrain()

                    [self.model, reference_state]= self.parameters['train_module'](X_retrain,Y_retrain,
                                                                          self.parameters['model']['model_type'],
                                                                          self.drift_detector, self.parameters['metric'])
                    self.drift_detector.setReferenceState(**reference_state)
                    self.drift_detector.setDefaultInitializeState()
                    flag_is_drift=True
                    self.buffer_X=[]
                    self.buffer_Y=[]
                    self.IntermediateStoreModel[evaluated_samples]=self.model # TODO make sorted dict

            if flag_is_drift:
                self.evaluation_engine.addRunningResults(drift = 1)
            else:
                self.evaluation_engine.addRunningResults(drift = 0)



    def BufferRetrain(self):
        # TODO: can be removed
            high_range= self.ilds.stream.current_timestamp+self.parameters['ilds']['chunk_size']
            low_range=int(high_range-(self.parameters['ilds']['chunk_size']*0.5))
            Y_oracle= self.oracle.getTrueLabelRange(low_range, high_range)
            X_retrain=X[len(X)-(high_range-low_range):]
            return [X_retrain_Y_oracle]


    def printResults(self, display_metrics=None):
      pp=pprint.PrettyPrinter()
      print 'Running Results Details'
      self.evaluation_engine.printSequentialMetrics()
      print 'Aggregated Results'
      pp.pprint(self.evaluation_engine.returnAggregates())
      print 'Number of drifts detected: %d' % self.evaluation_engine.returnNumberDrifts()
      print 'Expenditure by oracle: %d' % self.oracle.expenditure
      print 'Number of intermediate models made %d and time at:%s' %( len(self.IntermediateStoreModel.keys()), self.IntermediateStoreModel.keys())
      self.evaluation_engine.plotRunningTogether(display_metrics)


def LaunchPipeline(parameters):
    #   Stream
    ilds=data_reader.InitiallyLabeledDataStream(parameters['ilds']['filename'],
                                                  parameters['ilds']['initial_train_size'],
                                                  parameters['ilds']['chunk_size'],
                                                  parameters['ilds']['slide_rate'])

    # Oracle
    oracle=truoracle.Oracle(ilds.parent_dataset.test_set['Y'])

    # Drift Detection: Choose a drift detector to use here
    drift_detector=parameters['drift_detector']['method'](parameters=parameters['drift_detector']['parameters'])

    # Evaluation engine
    evaluation_engine=evaluate.Evaluation(attributes_to_track=parameters['evaluation']['attributes_to_track'])

    # Initial Model
    [model, reference_state]= parameters['train_module'](ilds.initial_labeled_set['X'],ilds.initial_labeled_set['Y'],\
                                            parameters['model']['model_type'], drift_detector, parameters['metric'])
    drift_detector.setReferenceState(**reference_state)

    ## Make Pipeline and start
    pipe = Pipeline(ilds=ilds, oracle=oracle, drift_detector=drift_detector, evaluation_engine=evaluation_engine,
                    train_module=parameters['train_module'], model =model, parameters=parameters)

    pipe.IntermediateStoreModel={0:model}

    pipe.Start()
    pipe.printResults(parameters['evaluation']['display_metrics'])
    return pipe.IntermediateStoreModel

def TestPipeline():

    parameters={}
    parameters['model']={'model_type': 'SVM_LINEAR'}
    parameters['metric']='ACCURACY'

    parameters['ilds']={}
    parameters['ilds']={
        'filename': '../../data/rh',
        'initial_train_size': 0.25,
        'chunk_size': 1000,
        'slide_rate': 100
    }

    parameters['drift_detector']={}
    parameters['drift_detector']['method']= global_constants.DRIFT_DETECTION_METHODS[1]
    parameters['drift_detector']['parameters']={
        'DROP_IN_ACCURACY': 0.1,
        'MIN_SAMPLES': 1000
    }
    parameters['drift_detector']['retrain_examples']=500
    parameters['train_module']=retrain_modules.retrain_module_ref_accuracy

    parameters['evaluation']={}
    parameters['evaluation']['attributes_to_track']=['drift', 'performance', 'performance_new', 'performance_sofar'] # defaults to performance and drift
    parameters['evaluation']['display_metrics']=['drift', 'performance']
    parameters['intermediate']=\
        {'print_counter': 1000}

    print 'Running Pipeline with parameters:'
    pp=pprint.PrettyPrinter()
    pp.pprint(parameters)

    LaunchPipeline(parameters)

if __name__=='__main__':
    TestPipeline()
