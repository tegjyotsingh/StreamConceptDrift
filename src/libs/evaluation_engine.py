__author__ = 'tegjyot'

import numpy as np
import matplotlib.pyplot as plt
from src.libs import global_constants

# keeps track of drift and performance statistics
class Evaluation(object):

    def __init__(self, attributes_to_track=None):
        # attributes to track are the metrics we need to provide is a string list
        if attributes_to_track is None:
            attributes_to_track=['drift', 'performance']
        self.attributes_to_track=attributes_to_track
        self.running_performance={}
        for metric in attributes_to_track:
            self.running_performance[metric]=[]


    def addRunningResults(self,**kwargs):
        # need to provide the attributes to track values as kwargs
        for key in kwargs:
            self.running_performance[key].append(kwargs[key])

    def returnNumberDrifts(self):
        return sum(self.running_performance['drift'])

    def returnAggregates(self):
        aggregatecounts={}
        for key in self.running_performance:
            aggregatecounts[key]={}
            aggregatecounts[key]['mean']=np.mean(self.running_performance[key])
            aggregatecounts[key]['std']=np.std(self.running_performance[key])
            aggregatecounts[key]['count']=len(self.running_performance[key])
        return aggregatecounts

    def printSequentialMetrics(self):
            X=np.arange(len(self.running_performance['drift']))
            for time in X:
                result=[]
                for key in self.attributes_to_track:
                    result.append(self.running_performance[key][time])
                print time, result


    def plotRunningTogether(self, display_metrics=None):
        if display_metrics is None:
            display_metrics=self.attributes_to_track
        X=np.arange(len(self.running_performance['drift']))
        legend=[]
        for i,metric in enumerate(display_metrics):
            if metric is 'drift' and self.returnNumberDrifts()!=0:
                [plt.axvline(x,color=global_constants.COLORS[i]) for (x,val) in enumerate(self.running_performance['drift']) if val==1]
            else:
                plt.plot(X,self.running_performance[metric],color=global_constants.COLORS[i])
            legend.append(metric)
            print i, metric
        plt.legend(legend)
        plt.xlabel('Time')
        plt.show()
