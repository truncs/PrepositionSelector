#! /usr/bin/python
import random
from numpy import *
from types import *

class Perceptron(object):
    """
    Implementation of the perceptron algorithm,
    which is  simple machine learning algorithm.
    """
    
    def __init__(self, event_size, outcome_size):
        assert event_size >=  1
        assert outcome_size >=  2
        self._count = 0
        self._predicted = 0
        self._learning_rate = random.random()
        self._weights = zeros(outcome_size, float64)
        self._input = zeros(event_size, float64)
        
    def learn(self, event, outcome):
        assert type(event) == ndarray 
        assert self._input.size  == event.size
        self._input = event
        pred = Perceptron._max_outcome(self.classify(event))
        if pred != outcome:
            self._update_weights(outcome, pred)
       
    def _update_weights(self, outcome, pred):
        error = outcome - pred
        for index in range(0,self._weights.size):
            self._weights[index] += self._learning_rate * error * self._input[index]

    @staticmethod
    def _max_outcome(pred_array):
        return pred_array[pred_array.argmax()]

    def classify(self, event):
        return self._weights * event
        
    def predict(self, outcome, event):
        self._count += 1
        pred = int(round(Perceptron._max_outcome(self.classify(event))))
        if pred == outcome:
            self._predicted += 1
            return True
        else:
            return False
        


        
            

        

        
            
        
        
        
