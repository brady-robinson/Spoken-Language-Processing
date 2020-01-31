import parselmouth
import glob
import os.path
import pandas as pd

# Citation: reading in .csv and analyzing soundwaves https://buildmedia.readthedocs.org/media/pdf/parselmouth/latest/parselmouth.pdf
timestamps_data = pd.read_csv("/Users/bradyrobinson/Desktop/slp_hmwrk_2/test_merged_timestamps.csv")

def get_prosody_features(row):
    output = []
    wav_code = row['helper']
    wav_code_clean = wav_code[0:6]

    start_col = timestamps_data['start_time']
    stop_col = timestamps_data['end_time']
    file_col = timestamps_data['helper']
    file_col_list = list(file_col)
    ind = file_col_list.index(wav_code)
    start = start_col[ind]
    stop = stop_col[ind]

    filepath = "/Users/bradyrobinson/Desktop/slp_hmwrk_2/wav_files/{}.wav".format(wav_code_clean)    
    sound = parselmouth.Sound(filepath)
    duration = sound.extract_part(start, stop)

    pitch = parselmouth.praat.call(duration, "To Pitch...", 0.0, 75.0, 600.0)
    min_pitch = parselmouth.praat.call(pitch, "Get minimum", 0.0, 0.0, "Hertz", "None")
    max_pitch = parselmouth.praat.call(pitch, "Get maximum", 0.0, 0.0, "Hertz", "None")
    range_pitch = max_pitch - min_pitch
    mean_pitch = parselmouth.praat.call(pitch, "Get mean", 0.0, 0.0, "Hertz")
    sd_pitch = parselmouth.praat.call(pitch, "Get standard deviation", 0.0, 0.0, "Hertz")

    energy = duration.get_energy()

    intensity = duration.get_intensity()

    output.append(min_pitch)
    output.append(max_pitch)
    output.append(range_pitch)
    output.append(mean_pitch)
    output.append(sd_pitch)
    output.append(energy)
    output.append(intensity)
    
    return output


# Citation: outputting to .csv https://buildmedia.readthedocs.org/media/pdf/parselmouth/latest/parselmouth.pdf
timestamps_data['prosody_features'] = timestamps_data.apply(get_prosody_features, axis='columns')

timestamps_data.to_csv("test_merged_timestamps_plus_speech_features.csv", index=False)
