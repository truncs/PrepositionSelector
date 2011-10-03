import pickle
from nltk import *
from numpy import *
from perceptron import *
from preposition_selector import *

infile = open('/home/aditya/nlp/training_bigrams', 'r')
bigram = pickle.load(infile)
bigram_dist = FreqDist(bigram)
bigram_prob_dist = WittenBellProbDist(bigram_dist, bigram_dist.__len__() + 10)

trigram_infile = open('/home/aditya/nlp/pos_trigrams', 'r')
trigram = pickle.load(trigram_infile)
trigram_dist = FreqDist(trigram)
trigram_prob_dist = WittenBellProbDist(trigram_dist, trigram_dist.__len__() + 10)


# Development data
in_dev_file = open('/home/aditya/nlp/in_dev', 'r')
in_dev = pickle.load(in_dev_file)
on_dev_file = open('/home/aditya/nlp/on_dev', 'r')
on_dev = pickle.load(on_dev_file)
of_dev_file = open('/home/aditya/nlp/of_dev', 'r')
of_dev = pickle.load(of_dev_file)
prepositions  = {}
prepositions['in'] = in_dev
prepositions['of'] = of_dev
prepositions['on'] = on_dev

p_selector = PrepositionSelector(prepositions, 3, bigram_prob_dist, trigram_prob_dist)
perceptron = Perceptron(len(prepositions), len(prepositions))

while True:
    outcome_and_sents = p_selector.get_outcome_and_sents()
    if outcome_and_sents:
        input = p_selector.get_feature_vector(outcome_and_sents[1])
        perceptron.learn(input, outcome_and_sents[0])
    else:
        break
print perceptron._weights    
#grammar = "NP: {<IN><DT>?<JJ>*<NN>}"

#Test Data
#in_test_file = open('/home/aditya/nlp/in_test', 'r')
#in_test = pickle.load(in_test_file)
#on_test_file = open('/home/aditya/nlp/on_test', 'r')
#on_test = pickle.load(on_test_file)
#of_test_file = open('/home/aditya/nlp/of_test', 'r')
#of_test = pickle.load(of_test_file)

