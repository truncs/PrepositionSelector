#! /usr/bin/python

import pickle
from nltk import *
from numpy import *
from types import *
import random

class PrepositionSelector(object):
    """ High level class for prepositon selector
    """
    
    def __init__(self, prepositions,preposition_data, bigram_prob, pos_trigrams_prob):
        """
        Loads the prepositions, initializes the outcomes
        and loads the data for features
        prepositions - dict of prepositions and the list of the associated data
        multiplier - multiplier for labels
        bigram_prob - bigrams probability distribution
        pos_trigrams_prob - pos trigrams probababilty distribution
        """
        self._bigram_prob = bigram_prob
        self._trigram_prob = pos_trigrams_prob
        self._prepositions = tuple(prepositions)
        self._preposition_data = preposition_data
        self._outcome_and_sents = []
        for key in self._preposition_data.keys():
            sentences = self._preposition_data[key]
            for sents in sentences:
                temp = []
                temp.append(self._prepositions.index(key))
                temp.append(sents)
                self._outcome_and_sents.append(temp)

    def get_feature_vector(self, sents) :
        vector = zeros(len(self._prepositions), float64)
        for counter in range(0, len(self._prepositions)):
            vector[counter] = self.calculate_feature_value(sents, self._prepositions[counter])
        return vector

    def get_outcome_and_sents(self):
        if len(self._outcome_and_sents) == 0:
            return False;
        else:
            return self._outcome_and_sents.pop(random.randrange(0, len(self._outcome_and_sents)))
        
    def calculate_feature_value(self, sents, prep):
        temp = list(sents)
        index = temp.index('*')
        
        temp[index] = prep
        bigram = bigrams(temp)
        bigram_left = tuple(PrepositionSelector.get_index(bigram,index -1))
        bigram_right = tuple(PrepositionSelector.get_index(bigram,index))

        pos_sents = self.get_pos_sents(temp, prep)
        trigram = trigrams(pos_sents)
        trigram_left = tuple(PrepositionSelector.get_index(trigram,index -2))
        trigram_right = tuple(PrepositionSelector.get_index(trigram, index))
        return self._bigram_prob.prob(bigram_left) + self._bigram_prob.prob(bigram_right) + self._trigram_prob.prob(trigram_left) + self._trigram_prob.prob(trigram_right)
        

    def get_pos_sents(self, sents, prep):
        prep_tokens = []
        pos_tokens = pos_tag(sents)
        for (word, pos) in pos_tokens:
            if(word == prep):
                prep_tokens.append(word + '|' + pos)
            else:
                prep_tokens.append(pos)

        return prep_tokens

    @staticmethod
    def get_index(a, index):
        """
       This functions takes a list  and index returns 
       the index if it is found otherwise returns an empty list
        Arguments:
        - `a`: list
        - `index`: index to be extracted from the list
        """
        if index < 0:
            return []
        element = []
        try:
            element = a[index]
        except:
            pass
        return element
        

    
