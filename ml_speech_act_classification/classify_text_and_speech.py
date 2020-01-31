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
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

# Citation: reading in .csv and analyzing soundwaves https://buildmedia.readthedocs.org/media/pdf/parselmouth/latest/parselmouth.pdf
train_data_text = pd.read_csv("/Users/bradyrobinson/Desktop/slp_hmwrk_2/train_merged_timestamps_plus_text_features_all_top_10.csv")
test_data_text = pd.read_csv("/Users/bradyrobinson/Desktop/slp_hmwrk_2/test_merged_timestamps_plus_text_features_all_top_10.csv")
train_data_speech = pd.read_csv("/Users/bradyrobinson/Desktop/slp_hmwrk_2/train_merged_timestamps_plus_speech_features_all_top_10.csv")
test_data_speech = pd.read_csv("/Users/bradyrobinson/Desktop/slp_hmwrk_2/test_merged_timestamps_plus_speech_features_all_top_10.csv")

train_data_text = train_data_text.drop(["act_tag"], axis=1)
test_data_text = test_data_text.drop(["act_tag"], axis=1)


col_names = ["min_pitch", "max_pitch", "range_pitch", "mean_pitch", "sd_pitch", "energy", "intensity", "speaking_rate",
            "number_words", "pos_sequence", "bigram_pos_counts", "bigrams", "bigram_counts", "compound_sentiment"]

le = preprocessing.LabelEncoder()
train_data_cat_fit = train_data_text.apply(le.fit_transform)
test_data_cat_fit = test_data_text.apply(le.fit_transform)

join_data_train = pd.merge(train_data_speech, train_data_cat_fit, right_index = True, left_index = True)
join_data_test = pd.merge(test_data_speech, test_data_cat_fit, right_index = True, left_index = True)

X_train = join_data_train[col_names]
y_train = join_data_train["act_tag"]

X_test = join_data_test[col_names]
y_test = join_data_test["act_tag"]

dtc_model = DecisionTreeClassifier()

dtc = dtc_model.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("F1: ", metrics.f1_score(y_test, y_pred, 
      labels=["sd", "b", "sv", "aa", "%", "ba", "qy", "x", "ny", "fc"], 
      average = "weighted"))
print(confusion_matrix(y_test, y_pred, 
      labels=["sd", "b", "sv", "aa", "%", "ba", "qy", "x", "ny", "fc"]))

# Citation: outputting to .csv https://buildmedia.readthedocs.org/media/pdf/parselmouth/latest/parselmouth.pdf
#text_data['text_features'] = text_data.apply(get_text_features, axis='columns')

#text_data.to_csv("test_merged_timestamps_plus_text_features.csv", index=False)
