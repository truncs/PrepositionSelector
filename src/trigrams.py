#! /usr/bin/python 

# Get the raw data and tokenize it 
# pos_tag it and store it in b 
# get the trigrams and store it in a file 


text1 = convote_training.raw()
pos_tokens = nltk.pos_tag(nltk.wordpunct_tokenize(text1))
trigram = nltk.trigram([d for (c,d) in pos_tokens])

trigram_file = open('/home/aditya/nlp/pos_trigrams', 'w')
pickle.dump(trigram, trigram_file)
