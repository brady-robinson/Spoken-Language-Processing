import nltk
import glob
import os.path
import pandas as pd
import string
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix



# Citation: reading in .csv and analyzing soundwaves https://buildmedia.readthedocs.org/media/pdf/parselmouth/latest/parselmouth.pdf
train_data = pd.read_csv("/Users/bradyrobinson/Desktop/slp_hmwrk_2/train_merged_timestamps_plus_speech_features_all_top_10.csv")

sd_counter = 0
b_counter = 0
sv_counter = 0
aa_counter = 0
perc_counter = 0
ba_counter = 0
qy_counter = 0
x_counter = 0
ny_counter = 0
fc_counter = 0

sd_total = 0
b_total = 0
sv_total = 0
aa_total = 0
perc_total = 0
ba_total = 0
qy_total = 0
x_total = 0
ny_total = 0
fc_total = 0

counter = 0

for num in train_data["speaking_rate"]:
      if train_data["act_tag"][counter] == "sd":
        sd_total += num
        sd_counter += 1
      elif train_data["act_tag"][counter] == "b":
        b_total += num
        b_counter += 1
      elif train_data["act_tag"][counter] == "sv":
        sv_total += num
        sv_counter += 1
      elif train_data["act_tag"][counter] == "aa":
        aa_total += num
        aa_counter += 1
      elif train_data["act_tag"][counter] == "%":
        perc_total += num
        perc_counter += 1
      elif train_data["act_tag"][counter] == "ba":
        ba_total += num
        ba_counter += 1
      elif train_data["act_tag"][counter] == "qy":
        qy_total += num
        qy_counter += 1
      elif train_data["act_tag"][counter] == "x":
        x_total += num
        x_counter += 1
      elif train_data["act_tag"][counter] == "ny":
        ny_total += num
        ny_counter += 1
      elif train_data["act_tag"][counter] == "fc":
        fc_total += num
        fc_counter += 1
      else:
        print("Error!")
      counter += 1


sd_mean = sd_total/sd_counter
b_mean = b_total/b_counter
sv_mean = sv_total/sv_counter
aa_mean = aa_total/aa_counter
perc_mean = perc_total/perc_counter
ba_mean = ba_total/ba_counter
qy_mean = qy_total/qy_counter
x_mean = x_total/x_counter
ny_mean = ny_total/ny_counter
fc_mean = fc_total/fc_counter

# Citation: bar chart https://www.youtube.com/watch?v=I653tWg0hCA

labels = ["sd", "b", "sv", "aa", "%", "ba", "qy", "x", "ny", "fc"]
values = [sd_mean, b_mean, sv_mean, aa_mean, perc_mean, ba_mean,
          qy_mean, x_mean, ny_mean, fc_mean]
order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

graphic = plt.figure(figsize=(5,5))
plt.bar(order, values)
plt.xticks(order, labels)
plt.title("Speaking Rate by Dialogue Act")
plt.xlabel("Dialogue Act")
plt.ylabel("Speaking Rate (words per second)")
plt.show()
