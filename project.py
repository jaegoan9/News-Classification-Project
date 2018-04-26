import pandas as pd
import numpy as np
import nltk
import re
import os.path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict

# nltk.download('stopwords')
# Initialize Stemmer
ps = PorterStemmer()

# Word match regex to exclude digits
word_pattern = re.compile(r'[^\W\d]')
# Set of stopwords
stopword_set = set(stopwords.words('english'))

# Import news title dataset with pandas
dataset = pd.read_csv('uci-news-aggregator.csv')

# List containing each title as item
titles = [title for title in dataset['TITLE']]

# Only create stemmed word file if no file exists
if (not os.path.exists('stemmed.txt') or not os.path.exists('stemmed_sentences.txt')):
	with open('stemmed.txt', 'w') as file, open('stemmed_sentences.txt', 'w') as file1:
		for title in titles:
			temp_title = []
			# split title into vector of words
			for word in title.split():
				if word.lower() not in stopword_set and 'http://' not in word:
					if word_pattern.match(word.lower()):
						# Add stemmed words to text file
						file.write(ps.stem(word.lower()) + '\n')
						temp_title.append(ps.stem(word.lower()))
			file1.write(' '.join(temp_title) + '\n')


# List of all stemmed words (bag of words)
all_words = []
with open('stemmed.txt', 'r') as file:
	for word in file:
		word = word.strip('\n')
		all_words.append(word)

# Contains word freq dictionary for each title
titles_doc_vector = []

with open('stemmed_sentences.txt', 'r') as file1:
	for title in file1:
		new_doc_vec = defaultdict(int)
		for word in title.split():
			new_doc_vec[word] += 1
		titles_doc_vector.append(new_doc_vec)

# Set of al unique stemmed words
vocab_set = list(set(all_words))

feature_vector = []
for freq_dict in titles_doc_vector:
	temp_feat = []
	for word in vocab_set:
		if word in freq_dict:
			temp_feat.append(freq_dict[word])
		else:
			temp_feat.append(0)
	feature_vector.append(np.asarray(temp_feat))
	break

print(feature_vector[0])

for i in feature_vector[0]:
	if i == 1:
		print("asd")