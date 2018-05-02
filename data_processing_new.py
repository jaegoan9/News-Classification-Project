from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import nltk
import sys
import re
import os.path
import unicodedata
import multiprocessing
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict, OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

nltk.download('stopwords')
nltk.download('wordnet')
# Initialize Stemmer
ps = PorterStemmer()
wl = WordNetLemmatizer()
# Word match regex to exclude digits
word_pattern = re.compile(r'[^\W\d]')
# Set of stopwords
stopword_set = set(stopwords.words('english'))

# Import news title dataset with pandas
dataset = pd.read_csv('uci-news-aggregator.csv')

# List containing each title as item
titles = [title for title in dataset['TITLE']]
true_label = [label for label in dataset['CATEGORY']]

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

selected_labels = []
# Only create stemmed word file if no file exists
if (not os.path.exists('lemmatized.txt') or not os.path.exists('lemmatized_sentences.txt')):
	print("Creating text files to extract from......")
	with open('lemmatized.txt', 'w') as file, open('lemmatized_sentences.txt', 'w') as file1:
		for ind, title in enumerate(titles):
			temp_title = []
			# split title into vector of words
			if "\xc2" not in title:
				selected_labels.append(true_label[ind])
				title = title.replace("\xe2", " ")
				for word in title.split():
					word = word.lower().rstrip('?:!.,;')
					word = clean_str(word)
					word = ''.join(c for c in word if c.isalpha())
					if word not in stopword_set and 'http://' not in word and 'www' not in word:
						if word_pattern.match(word):
							# Add stemmed words to text file
							try:
								# word_stem = ps.stem(word).rstrip("'")
								word_temp = wl.lemmatize(word).rstrip("'")
								file.write(word_temp + '\n')
								temp_title.append(word_temp)
							except UnicodeDecodeError:
								print(word)
								# print(ps.stem(word).rstrip("'"))
								print("=================\n")
				# Write titles to stemmed_sentences.txt
				file1.write(' '.join(temp_title) + '\n')
else:
	print ("lemmanized text files already present.")

# List of all stemmed words (bag of words)
all_words = []
corpus_freq = defaultdict(int)
feature = OrderedDict()
with open('lemmatized.txt', 'r') as file:
	for word in file:
		word = word.strip('\n')
		all_words.append(word)
		corpus_freq[word] += 1.0
		feature[word] = 0

# Contains word freq dictionary for each title
titles_doc_vector = []

with open('lemmatized_sentences.txt', 'r') as file1:
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

freq_descending = sorted(freq_count, reverse=True)

#removing unnecessary
for key, val in corpus_freq.items():
    if (val >= 0 and val <= 20) or val in freq_descending[0:200]:
        del corpus_freq[key]
        del feature[key]
print('\n')
print('finished processing')
print("length of features is " +str(len(feature)))
print("length of features is " +str(len(corpus_freq)))
print("length of examples is " + str(len(titles_doc_vector)))

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
	p = multiprocessing.Process(target=run_thread, args=(titles_doc_vector[(i-1)*20000:i*20000], feature, (i-1)*20000))
	jobs.append(p)
	p.start()