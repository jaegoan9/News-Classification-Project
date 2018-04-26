import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict

# nltk.download('stopwords')
# Initialize Stemmer
ps = PorterStemmer()

# Set of stopwords
stopword_set = set(stopwords.words('english'))

# Import news title dataset with pandas
dataset = pd.read_csv('uci-news-aggregator.csv')

titles = [title for title in dataset['TITLE']]

all_words = []
with open('stemmed.txt', 'w') as file:
	for title in titles:
		for word in title.split():
			if word not in stopword_set and 'http://' not in word:
				file.write(word.lower() + '\n')
				all_words.append(word.lower().replace('\'', ''))

word_freq = defaultdict(int)
vocab_set = set(all_words)

for word in all_words:
	word_freq[word] += 1

# print(vocab_set)
print(len(vocab_set))
