import pandas as pd
import numpy as np
import nltk
import sys
import re
import os.path
import multiprocessing
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict, OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# List of all stemmed words (bag of words)
all_words = []
corpus_freq = defaultdict(int)
feature = OrderedDict()
with open('stemmed.txt', 'r') as file:
	for word in file:
		word = word.strip('\n')
		all_words.append(word)
		corpus_freq[word] += 1.0
		feature[word] = 0

# Contains word freq dictionary for each title
titles_doc_vector = []

with open('stemmed_sentences.txt', 'r') as file1:
	for title in file1:
		new_doc_vec = defaultdict(int)
		for word in title.split():
			new_doc_vec[word] += 1
		titles_doc_vector.append(new_doc_vec)

# Set of all unique stemmed words
vocab_set = list(set(all_words))
print("length of all words (including repeats) is: " + str(len(all_words)))
print("length of vocab list is " +str(len(vocab_set)))
print("length of titles list is " + str(len(titles_doc_vector)))

freq_count = defaultdict(int)
for key, val in corpus_freq.items():
    freq_count[val] += 1
print("Number of words with freq = 1: " + str(freq_count[1]))
print("Number of words with freq = 2: " + str(freq_count[2]))
print("Number of words with freq = 3: " + str(freq_count[3]))
print("Number of words with freq = 4: " + str(freq_count[4]))
print("Number of words with freq = 5: " + str(freq_count[5]))
print("Number of words with freq = 6: " + str(freq_count[6]))
print("Number of words with freq = 7: " + str(freq_count[7]))
print("Number of words with freq = 8: " + str(freq_count[8]))
print("Number of words with freq = 9: " + str(freq_count[9]))
print("Number of words with freq = 10: " + str(freq_count[10]))

#removing unnecessary
for key, val in corpus_freq.items():
    if val >= 0 and val <= 10:
        del corpus_freq[key]
        del feature[key]

def run_thread(doc_vector, feature, start):
	feature_vector = OrderedDict()
	counter = 0
	print("start :" + str(start))
	for freq_dict in doc_vector:
	    counter += 1
	    instance = feature.copy()
	    for word, freq in freq_dict.items():
	        if word in corpus_freq and corpus_freq[word] != 0:
	            instance[word] = freq / corpus_freq[word]
	    feature_vector[counter] = instance.values()
	    if counter == 1000:
	        start += 1
	        with open('feature_json_multi/feature_vector' + str(start) + '.json', 'w') as fp:
	            json.dump(feature_vector, fp)
	            feature_vector.clear()
	            print('finished ' + str(start) + ' json file')
	            counter = 0

jobs = []
for i in range(1,5):
	p = multiprocessing.Process(target=run_thread, args=(titles_doc_vector[(i-1)*50000:i*50000], feature, i*50000))
	jobs.append(p)
	p.start()