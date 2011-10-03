#! /usr/bin/python

import pickle
from nltk import *
from nltk.corpus import PlaintextCorpusReader

training_corpus_root = "../convote/data_stage_one/training_set"
dev_corupus_root = "../convote/data_stage_one/development_set"
test_corupus_root = "../convote/data_stage_one/test_set"
convote_training = PlaintextCorpusReader(training_corpus_root, '.*')
convote_dev = PlaintextCorpusReader(dev_corupus_root, '.*')
convote_test = PlaintextCorpusReader(test_corupus_root, '.*')
 
bigram = bigrams(convote_training.tokenized())
outfile = open('/home/aditya/nlp/training_bigrams', 'w')

pos_tokens = pos_tag(convote_training.tokenized())
prep_tokens = []

for (word, pos) in pos_tokens:
    if(pos == 'IN'):
        prep_tokens.append(word + '|' + pos)
    else:
        prep_tokens.append(pos)
        
trigram = nltk.trigrams(prep_tokens)
trigram_file = open('/home/aditya/nlp/pos_trigrams', 'w')

pickle.dump(bigram, outfile)
pickle.dump(trigram, trigram_file)
    
for sents in convote_test.sents():
    for index in range(0, len(sents)):
        if sents[index] == 'in':
            temp = deepcopy(sents)
            temp[index] = '*'
            in_test.append(temp)
        if sents[index] == 'on':
            temp = deepcopy(sents)
            temp[index] = "*"
            on_test.append(temp)
        if sents[index] == 'of':
            temp = deepcopy(sents)
            temp[index] = "*"
            of_test.append(temp)

for sents in convote_dev.sents():
    for index in range(0, len(sents)):
        if sents[index] == 'in':
            temp = deepcopy(sents)
            temp[index] = '*'
            in_dev.append(temp)
        if sents[index] == 'on':
            temp = deepcopy(sents)
            temp[index] = "*"
            on_dev.append(temp)
        if sents[index] == 'of':
            temp = deepcopy(sents)
            temp[index] = "*"
            of_dev.append(temp)

def deepcopy(s):
    picklestring = pickle.dumps(s)
    return pickle.loads(picklestring)
