import librosa.display

import matplotlib.pyplot as plt
import numpy as np

import dataset as d
import wav_to_image as wti

data_folder_path = './data/'
dataset = d.SoundData(data_folder_path)
resample_rate = 8000
dataset.resample_rate = resample_rate

second = 6
dataset.max_second = second

for idx, data in enumerate(dataset):
    file_name = data['file_name']
    sig = data['waveform']

    # STFT -> spectrogram
    hop_length = 512  # 전체 frame 수
    # display spectrogram
    plt.figure(figsize=(10, 5))
    mel = librosa.feature.melspectrogram(y=sig, sr=resample_rate)
    librosa.display.specshow(mel, sr=resample_rate, hop_length=hop_length)
    plt.savefig(f'./{file_name}_1.png', bbox_inches="tight", pad_inches=0)
    plt.plot()
    plt.close()
    if idx == 4:
        break
