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


# Development data
in_dev_file = open('../data/in_dev', 'r')
in_dev = pickle.load(in_dev_file)
on_dev_file = open('../data/on_dev', 'r')
on_dev = pickle.load(on_dev_file)
of_dev_file = open('../data/of_dev', 'r')
of_dev = pickle.load(of_dev_file)
preposition_data  = {}
preposition_data['in'] = in_dev
preposition_data['of'] = of_dev
preposition_data['on'] = on_dev
prepositions = ('in', 'of', 'on')
p_selector = PrepositionSelector(prepositions, preposition_data, bigram_prob_dist, trigram_prob_dist)
perceptron = Perceptron(prepositions,len(prepositions), len(prepositions))

while True:
    outcome_and_sents = p_selector.get_outcome_and_sents()
    if outcome_and_sents:
        event = p_selector.get_feature_vector(outcome_and_sents[1])
        perceptron.predict(event, outcome_and_sents[0], True)
        perceptron._count += 1
    else:
        break
print perceptron._weights    
#grammar = "NP: {<IN><DT>?<JJ>*<NN>}"
weight_file = open('weights', 'w')
pickle.dump(perceptron._weights, weight_file)


