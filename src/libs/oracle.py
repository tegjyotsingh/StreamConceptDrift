__author__ = 'tegjyot'

# class to define oracle for active labeling

class Oracle(object):
    def __init__(self,Y):
        self.ground_truth=Y
        self.expenditure=0

    def getTrueLabelEvaluation(self,id):
        return self.ground_truth[id]

    def getTrueLabel(self,id, cost=1):
        self.expenditure+=cost
        return self.ground_truth[id]

    def getTrueLabelRange(self, id_low, id_high, cost=None):
        if cost is None:
            cost=id_high-id_low
        self.expenditure+=cost
        return [self.ground_truth[id] for id in range(id_low,id_high)]

