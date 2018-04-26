import pandas as pd
import numpy as np
import nltk
import re
import os.path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

nltk.download('stopwords')
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
true_label = [label for label in dataset['CATEGORY']]

# Only create stemmed word file if no file exists
if (not os.path.exists('stemmed.txt') or not os.path.exists('stemmed_sentences.txt')):
	print("Creating text files to extract from......")
	with open('stemmed.txt', 'w') as file, open('stemmed_sentences.txt', 'w') as file1:
		for title in titles:
			temp_title = []
			# split title into vector of words
			for word in title.split():
				word = word.lower().rstrip('?:!.,;')
				if word not in stopword_set and 'http://' not in word and 'www' not in word:
					if word_pattern.match(word):
						# Add stemmed words to text file
						word_temp = ps.stem(word).rstrip("'")
						file.write(word_temp + '\n')
						temp_title.append(word_temp)
			# Write titles to stemmed_sentences.txt
			file1.write(' '.join(temp_title) + '\n')
else:
	print ("stemmed text files already present.")

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

# Set of all unique stemmed words
vocab_set = list(set(all_words))
print("length of all words (including repeats) is: " + str(len(all_words)))
print("length of vocab list is " +str(len(vocab_set)))
print("length of titles list is " + str(len(titles_doc_vector)))

# Feature vector of length number of titles
feature_vector = []
counter = 0
with open('feature_vector.txt', 'w') as file2:
	for freq_dict in titles_doc_vector:
		counter += 1
		temp_feat = []
		for word in vocab_set:
			if word in freq_dict:
				temp_feat.append(freq_dict[word])
			else:
				temp_feat.append(0)
		feature_vector.append(np.asarray(temp_feat))
	# This part takes really damn long
		# if(counter == 10000):  # This is for testing purposes
		# 	break

print("two")
# Decision Tree Classifier trial
train, test, train_labels, test_labels = train_test_split(feature_vector,
                                                      true_label,
                                                      test_size=0.33) #do a train_test split
model = DecisionTreeClassifier()
model.fit(train, train_labels)
preds = model.predict(test)

print ('%s %d %s %.3f %s %s %d %s %.3f %s' % ("### OVERALL CORRECT: ", 
accuracy_score(test_labels, preds) * len(test_labels), " = ",  
accuracy_score(test_labels, preds) * 100, "%   ", "INCORRECT: ", 
len(test_labels) - accuracy_score(test_labels, preds)*len(test_labels), " = ",  
100 - accuracy_score(test_labels, preds) * 100, "%")) #print out results