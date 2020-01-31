import nltk
import glob
import os.path
import pandas as pd
import string
import numpy
import math
from nltk.util import bigrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sent_an = SentimentIntensityAnalyzer()


# Citation: reading in .csv and analyzing soundwaves https://buildmedia.readthedocs.org/media/pdf/parselmouth/latest/parselmouth.pdf
text_data = pd.read_csv("/Users/bradyrobinson/Desktop/slp_hmwrk_2/test_merged_timestamps.csv")

def get_text_features(row):
    output = []
    pos_tags_array = []
    bis = ""
    bis_for_freq = ""
    bis_freq_counts = {}
    compound_sent = ""
    tok_pos_tags = ""
    bis_tok_pos_tags = ""
    bis_pos_freq_counts = {}
    wav_code = row['helper']
    wav_code_clean = wav_code[0:6]

    text_col = text_data['clean_text']
    
    file_col = text_data['helper']
    file_col_list = list(file_col)
    ind = file_col_list.index(wav_code)
    current_utt = text_col[ind]

# Citation: tokenize http://www.nltk.org/api/nltk.html
    try:
        toks_current_utt = nltk.word_tokenize(current_utt)
        pos_tags = nltk.pos_tag(toks_current_utt)
        for tup in pos_tags:
            pos_tags_array.append(tup[1])
        length_current_utt = len(toks_current_utt)
    except TypeError:
        length_current_utt = 0

    bis_tok_pos_tags = list(nltk.bigrams(pos_tags_array))
    for bigr in bis_tok_pos_tags:
        if bigr not in list(bis_pos_freq_counts.keys()):
            bis_pos_freq_counts[bigr] = 1
        else:
            bis_pos_freq_counts[bigr] += 1

    try:
        toks_current_utt = nltk.word_tokenize(current_utt)
        bis = list(nltk.bigrams(toks_current_utt))
    except TypeError:
        ""

# Citation: bigrams http://www.nltk.org/api/nltk.html
    try:
        toks_current_utt = nltk.word_tokenize(current_utt)
        bis_for_freq = list(nltk.bigrams(toks_current_utt))
        for bi in bis_for_freq:
            if bi not in list(bis_freq_counts.keys()):
                bis_freq_counts[bi] = 1
            else:
                bis_freq_counts[bi] += 1
    except TypeError:
        ""
# Citation: sentiment analysis: http://www.nltk.org/howto/sentiment.html
    try:
        sent_scores = sent_an.polarity_scores(current_utt)
        compound_sent = sent_scores['compound']
    except AttributeError:
        ""

    output.append(length_current_utt)
    output.append("|")
    output.append(pos_tags_array)
    output.append("|")
    output.append(bis_pos_freq_counts)
    output.append("|")
    output.append(bis)
    output.append("|")
    output.append(bis_freq_counts)
    output.append("|")
    output.append(compound_sent)
    
    return output

# Citation: outputting to .csv https://buildmedia.readthedocs.org/media/pdf/parselmouth/latest/parselmouth.pdf
text_data['text_features'] = text_data.apply(get_text_features, axis='columns')

text_data.to_csv("test_merged_timestamps_plus_text_features.csv", index=False)
