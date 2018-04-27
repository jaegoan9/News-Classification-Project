import pandas as pd
import numpy as np
import nltk
import sys
import re
import os.path
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict, OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

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
print('\n')
print('finished processing')
print("length of features is " +str(len(feature)))
print("length of features is " +str(len(corpus_freq)))
print("length of examples is " + str(len(titles_doc_vector)))

# Feature vector of length number of titles
# feature_vector = []
# counter = 0
# with open('feature_vector.txt', 'w') as file2:
# 	for freq_dict in titles_doc_vector:
# 		counter += 1
# 		temp_feat = []
# 		for word in vocab_set:
# 			if word in freq_dict:
# 				temp_feat.append(freq_dict[word] / corpus_freq[word])
# 			else:
# 				temp_feat.append(0)
# 		feature_vector.append(np.asarray(temp_feat))
# 	# This part takes really damn long
# 		if(counter == 10000):  # This is for testing purposes
# 			break
# 		# Change "true_label" in train test split below to true_label[:10000]
# 		# for segment testing
# 		# Program WILL CRASH if run on full feature vector under current setting

#load json file
data_vector = []
for i in range(1, 21):
    data_json = json.load(open('feature_json/feature_vector' + str(i) + '.json'), object_pairs_hook=OrderedDict)
    data_vector += data_json.values()
print("Completely loaded data")

# Decision Tree Classifier trial
train, test, train_labels, test_labels = train_test_split(data_vector,
                                                      true_label[:20000],
                                                      test_size=0.33) #do a train_test split
model = DecisionTreeClassifier()
model.fit(train, train_labels)
preds = model.predict(test)

print ('%s %d %s %.3f %s %s %d %s %.3f %s' % ("### OVERALL CORRECT: ", 
accuracy_score(test_labels, preds) * len(test_labels), " = ",  
accuracy_score(test_labels, preds) * 100, "%   ", "INCORRECT: ", 
len(test_labels) - accuracy_score(test_labels, preds)*len(test_labels), " = ",  
100 - accuracy_score(test_labels, preds) * 100, "%")) #print out results

#Naive Bayes
model = MultinomialNB()
y_pred = model.fit(train, train_labels).predict(test)
print ('%s %d %s %.3f %s %s %d %s %.3f %s' % ("### MULTINOMIALNB OVERALL CORRECT: ", 
accuracy_score(test_labels, y_pred) * len(test_labels), " = ",  
accuracy_score(test_labels, y_pred) * 100, "%   ", "INCORRECT: ", 
len(test_labels) - accuracy_score(test_labels, y_pred)*len(test_labels), " = ",  
100 - accuracy_score(test_labels, y_pred) * 100, "%")) #print out results

#SVM
model = SVC()
y_pred = model.fit(train, train_labels).predict(test)
print ('%s %d %s %.3f %s %s %d %s %.3f %s' % ("### SVM OVERALL CORRECT: ", 
accuracy_score(test_labels, y_pred) * len(test_labels), " = ",  
accuracy_score(test_labels, y_pred) * 100, "%   ", "INCORRECT: ", 
len(test_labels) - accuracy_score(test_labels, y_pred)*len(test_labels), " = ",  
100 - accuracy_score(test_labels, y_pred) * 100, "%")) #print out results


