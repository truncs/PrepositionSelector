#! /usr/bin/python
import random
from numpy import *
from types import *

class Perceptron(object):
    """
    Implementation of the perceptron algorithm,
    which is a  simple machine learning algorithm.
    """
    
    def __init__(self, input_vector,event_size, outcome_size,weights=None):
        assert event_size >=  1
        assert outcome_size >=  2
        self._count = 0
        self._predicted = 0
        self._learning_rate = random.random()
        if weights is None:
            self._weights = zeros(outcome_size, float64)
        else:
            assert type(weights) == ndarray
            self._weights = weights
        self._input = tuple(input_vector)
        self._incorrect_predictions = []
        
    def predict(self, event, outcome, learn=False):
        ''''
        calculate the predicted outcome and compare it with
        the actual outcome, and then update the weights
        - `event`: vector in inputs
        - `outcome`: index of the outcome
        - `learn`: bool for learning mode
        '''
        assert type(event) == ndarray 
        assert len(self._input)  == event.size
        self._count += 1
        outcome_vector = self.classify(event)
        pred = outcome_vector.argmax()
        if pred != outcome:
            if learn:
                self._update_weights(outcome, event, 1 )
                self._update_weights(pred, event, -1)
            return (False, pred)
        else:
            self._predicted += 1
            return (True, pred)
        
    def _update_weights(self, index, event, delta):
        self._weights[index] += event[index]*self._learning_rate*delta

    def classify(self, event):
        return self._weights * event
        
    def accuracy(self):
        return float(self._predicted) / self._count


        
            

        

        
            
        
        
        
