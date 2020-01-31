# input: a sound file, timestamps, and utterance numbers
# output: acoustic-prosodic features for the given row

import nltk
import glob
import os.path
import pandas as pd
import string
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing



# Citation: reading in .csv and analyzing soundwaves https://buildmedia.readthedocs.org/media/pdf/parselmouth/latest/parselmouth.pdf
train_data = pd.read_csv("/Users/bradyrobinson/Desktop/slp_hmwrk_2/train_merged_timestamps_plus_text_features_all_top_10.csv")
test_data = pd.read_csv("/Users/bradyrobinson/Desktop/slp_hmwrk_2/test_merged_timestamps_plus_text_features_all_top_10.csv")

col_names = ["number_words", "pos_sequence", "bigram_pos_counts", "bigrams", "bigram_counts", "compound_sentiment"]

# citation: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
# citation: https://towardsdatascience.com/scikit-learn-decision-trees-explained-803f3812290d
# citation: http://benalexkeen.com/decision-tree-classifier-in-python-using-scikit-learn/

le = preprocessing.LabelEncoder()
train_data_cat_fit = train_data.apply(le.fit_transform)
test_data_cat_fit = test_data.apply(le.fit_transform)

X_train = train_data_cat_fit[col_names]
y_train = train_data["act_tag"]

X_test = test_data_cat_fit[col_names]
y_test = test_data["act_tag"]

dtc_model = DecisionTreeClassifier()

dtc = dtc_model.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("F1: ", metrics.f1_score(y_test, y_pred, 
      labels=["sd", "b", "sv", "aa", "%", "ba", "qy", "x", "ny", "fc"], 
      average = "weighted"))
print(confusion_matrix(y_test, y_pred, labels=["sd", "b", "sv", "aa", "%", "ba", "qy", "x", "ny", "fc"]))
