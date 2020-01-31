import parselmouth
import glob
import os.path
import pandas as pd

# Citation: reading in .csv and analyzing soundwaves https://buildmedia.readthedocs.org/media/pdf/parselmouth/latest/parselmouth.pdf
msp_data = pd.read_csv("/Users/bradyrobinson/Desktop/slp_hmwrk_1/msp_features.csv")
my_data = pd.read_csv("/Users/bradyrobinson/Desktop/slp_hmwrk_1/my_features.csv")

# Feature extraction methods
def get_min_pitch(row):
    speech_file = row['Speech_File']
    filepath = "/Users/bradyrobinson/Desktop/slp_hmwrk_1/MSP_samples/{}.wav".format(speech_file)
    sound = parselmouth.Sound(filepath)
    pitch = parselmouth.praat.call(sound, "To Pitch...", 0.0, 75.0, 600.0)
    min_pitch = parselmouth.praat.call(pitch, "Get minimum", 0.0, 0.0, "Hertz", "None")
    return min_pitch

def get_max_pitch(row):
    speech_file = row['Speech_File']
    filepath = "/Users/bradyrobinson/Desktop/slp_hmwrk_1/MSP_samples/{}.wav".format(speech_file)
    sound = parselmouth.Sound(filepath)
    pitch = parselmouth.praat.call(sound, "To Pitch...", 0.0, 75.0, 600.0)
    max_pitch = parselmouth.praat.call(pitch, "Get maximum", 0.0, 0.0, "Hertz", "None")
    return max_pitch

def get_mean_pitch(row):
    speech_file = row['Speech_File']
    filepath = "/Users/bradyrobinson/Desktop/slp_hmwrk_1/MSP_samples/{}.wav".format(speech_file)
    sound = parselmouth.Sound(filepath)
    pitch = parselmouth.praat.call(sound, "To Pitch...", 0.0, 75.0, 600.0)
    mean_pitch = parselmouth.praat.call(pitch, "Get mean", 0.0, 0.0, "Hertz")
    return mean_pitch

def get_sd_pitch(row):
    speech_file = row['Speech_File']
    filepath = "/Users/bradyrobinson/Desktop/slp_hmwrk_1/MSP_samples/{}.wav".format(speech_file)
    sound = parselmouth.Sound(filepath)
    pitch = parselmouth.praat.call(sound, "To Pitch...", 0.0, 75.0, 600.0)
    sd_pitch = parselmouth.praat.call(pitch, "Get standard deviation", 0.0, 0.0, "Hertz")
    return sd_pitch

def get_min_intensity(row):
    speech_file = row['Speech_File']
    filepath = "/Users/bradyrobinson/Desktop/slp_hmwrk_1/MSP_samples/{}.wav".format(speech_file)
    sound = parselmouth.Sound(filepath)
    intensity = sound.to_intensity()
    min_intensity = parselmouth.praat.call(intensity, "Get minimum", 0.0, 0.0, "None")
    return min_intensity

def get_max_intensity(row):
    speech_file = row['Speech_File']
    filepath = "/Users/bradyrobinson/Desktop/slp_hmwrk_1/MSP_samples/{}.wav".format(speech_file)
    sound = parselmouth.Sound(filepath)
    intensity = sound.to_intensity()
    max_intensity = parselmouth.praat.call(intensity, "Get maximum", 0.0, 0.0, "None")
    return max_intensity

def get_mean_intensity(row):
    speech_file = row['Speech_File']
    filepath = "/Users/bradyrobinson/Desktop/slp_hmwrk_1/MSP_samples/{}.wav".format(speech_file)
    sound = parselmouth.Sound(filepath)
    intensity = sound.to_intensity()
    mean_intensity = parselmouth.praat.call(intensity, "Get mean", 0.0, 0.0)
    return mean_intensity

def get_sd_intensity(row):
    speech_file = row['Speech_File']
    filepath = "/Users/bradyrobinson/Desktop/slp_hmwrk_1/MSP_samples/{}.wav".format(speech_file)
    sound = parselmouth.Sound(filepath)
    intensity = sound.to_intensity()
    sd_intensity = parselmouth.praat.call(intensity, "Get standard deviation", 0.0, 0.0)
    return sd_intensity

def get_speaking_rate(row):
    speech_file = row['Speech_File']
    filepath = "/Users/bradyrobinson/Desktop/slp_hmwrk_1/MSP_samples/{}.wav".format(speech_file)
    sound = parselmouth.Sound(filepath)
    total_duration = parselmouth.praat.call(sound, "Get total duration")
    if speech_file == "Angry":
      number_of_words = 13
    elif speech_file == "Disgust":
      number_of_words = 25
    elif speech_file == "Fear":
      number_of_words = 31
    elif speech_file == "Happy":
      number_of_words = 18
    elif speech_file == "Neutral":
      number_of_words = 10
    elif speech_file == "Sad":
      number_of_words = 16
    elif speech_file == "Surprise":
      number_of_words = 13
    elif speech_file == "my_Angry" or "my_Happy" or "my_Disgust" or "my_Neutral" or "my_Sad" or "my_Surprise" or "my_Fear":
      number_of_words = 10
    speaking_rate = number_of_words/total_duration
    return speaking_rate

def get_jitter(row):
    speech_file = row['Speech_File']
    filepath = "/Users/bradyrobinson/Desktop/slp_hmwrk_1/MSP_samples/{}.wav".format(speech_file)
    sound = parselmouth.Sound(filepath)
    pitch = sound.to_pitch()
    point_process = parselmouth.praat.call(pitch, "To PointProcess")
    jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    return jitter_local

# Citation: shimmer extraction https://stackoverflow.com/questions/54707734/parselmouth-batch-full-voice-report
def get_shimmer(row):
    speech_file = row['Speech_File']
    filepath = "/Users/bradyrobinson/Desktop/slp_hmwrk_1/MSP_samples/{}.wav".format(speech_file)
    sound = parselmouth.Sound(filepath)
    pitch = sound.to_pitch()
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    shimmer_local = parselmouth.praat.call([sound, pulses], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    return shimmer_local

# Citation: HNR https://buildmedia.readthedocs.org/media/pdf/parselmouth/latest/parselmouth.pdf
def get_hnr(row):
    speech_file = row['Speech_File']
    filepath = "/Users/bradyrobinson/Desktop/slp_hmwrk_1/MSP_samples/{}.wav".format(speech_file)
    sound = parselmouth.Sound(filepath)
    harmonicity = sound.to_harmonicity()
    harmonicity_cc = parselmouth.praat.call(sound, "To Harmonicity (cc)...", 0.01, 75.0, 0.1, 1.0)
    hnr = harmonicity_cc.values[harmonicity.values != -200].mean()
    return hnr

# Alter imported data frame containing msp podcast speech
# Citation: outputting to .csv https://buildmedia.readthedocs.org/media/pdf/parselmouth/latest/parselmouth.pdf
msp_data['Min_Pitch'] = msp_data.apply(get_min_pitch, axis='columns')
msp_data['Max_Pitch'] = msp_data.apply(get_max_pitch, axis='columns')
msp_data['Mean_Pitch'] = msp_data.apply(get_mean_pitch, axis='columns')
msp_data['Sd_Pitch'] = msp_data.apply(get_sd_pitch, axis='columns')
msp_data['Min_Inensity'] = msp_data.apply(get_min_intensity, axis='columns')
msp_data['Max_Inensity'] = msp_data.apply(get_max_intensity, axis='columns')
msp_data['Mean_Inensity'] = msp_data.apply(get_mean_intensity, axis='columns')
msp_data['Sd_Inensity'] = msp_data.apply(get_sd_intensity, axis='columns')
msp_data['Speaking_Rate'] = msp_data.apply(get_speaking_rate, axis='columns')
msp_data['Jitter'] = msp_data.apply(get_jitter, axis='columns')
msp_data['Shimmer'] = msp_data.apply(get_shimmer, axis='columns')
msp_data['HNR'] = msp_data.apply(get_hnr, axis='columns')

# Alter imported data frame containing my speech
my_data['Min_Pitch'] = my_data.apply(get_min_pitch, axis='columns')
my_data['Max_Pitch'] = my_data.apply(get_max_pitch, axis='columns')
my_data['Mean_Pitch'] = my_data.apply(get_mean_pitch, axis='columns')
my_data['Sd_Pitch'] = my_data.apply(get_sd_pitch, axis='columns')
my_data['Min_Inensity'] = my_data.apply(get_min_intensity, axis='columns')
my_data['Max_Inensity'] = my_data.apply(get_max_intensity, axis='columns')
my_data['Mean_Inensity'] = my_data.apply(get_mean_intensity, axis='columns')
my_data['Sd_Inensity'] = my_data.apply(get_sd_intensity, axis='columns')
my_data['Speaking_Rate'] = my_data.apply(get_speaking_rate, axis='columns')
my_data['Jitter'] = my_data.apply(get_jitter, axis='columns')
my_data['Shimmer'] = my_data.apply(get_shimmer, axis='columns')
my_data['HNR'] = my_data.apply(get_hnr, axis='columns')

# Output to .csv
msp_data.to_csv("msp_features_extracted.csv", index=False)
my_data.to_csv("my_features_extracted.csv", index=False)