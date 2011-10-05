import pickle
from nltk import *
from numpy import *
from perceptron import *
from preposition_selector import *

infile = open('../data/training_bigrams', 'r')
bigram = pickle.load(infile)
bigram_dist = FreqDist(bigram)
bigram_prob_dist = WittenBellProbDist(bigram_dist, bigram_dist.__len__() + 10)

trigram_infile = open('../data/pos_trigrams', 'r')
trigram = pickle.load(trigram_infile)
trigram_dist = FreqDist(trigram)
trigram_prob_dist = WittenBellProbDist(trigram_dist, trigram_dist.__len__() + 10)

in_test_file = open('../data/in_test', 'r')
in_test = pickle.load(in_test_file)
on_test_file = open('../data/on_test', 'r')
on_test = pickle.load(on_test_file)
of_test_file = open('../data/of_test', 'r')
of_test = pickle.load(of_test_file)

test_output = open('../data/test_output.txt', 'w')
weight_file = open('../data/weights', 'r')
weights = pickle.load(weight_file)

preposition_data  = {}
preposition_data['in'] = in_test
preposition_data['of'] = of_test
preposition_data['on'] = on_test
prepositions = ('in', 'of', 'on')
p_selector = PrepositionSelector(prepositions, preposition_data, bigram_prob_dist, trigram_prob_dist)
perceptron = Perceptron(prepositions,len(prepositions), len(prepositions), weights)
output_text = []
while True:
    outcome_and_sents = p_selector.get_outcome_and_sents()
    if outcome_and_sents:
        event = p_selector.get_feature_vector(outcome_and_sents[1])
        (x,y) =  perceptron.predict(event, outcome_and_sents[0], False)
        if x is not True:
            value = (y,  outcome_and_sents[0], outcome_and_sents[1])
            output_text.append(value)
    else:
        break

print perceptron._count
print perceptron._predicted
print perceptron.accuracy() * 100
        
