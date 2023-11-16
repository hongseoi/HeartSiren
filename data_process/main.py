import librosa.display

import matplotlib.pyplot as plt
import numpy as np

import dataset as d
import wav_to_image as wti

data_folder_path = '../../data_1107/'
dataset = d.SoundData(data_folder_path)
resample_rate = 8000
dataset.resample_rate = resample_rate

second = 5
dataset.max_second = second

for idx, data in enumerate(dataset):
    file_name = data['file_name']
    sig = data['waveform']

    # STFT -> spectrogram
    hop_length = 512
    # number of samples between successive frames. See librosa.core.stft
    frame_length = 0.064
    frame_stride = 0.025
    input_nfft = int(round(resample_rate * frame_length))
    input_stride = int(round(resample_rate * frame_stride))
    print(input_stride, input_nfft)
    hop_length = 1024
    # display spectrogram
    plt.figure(figsize=(10, 5))
    mel = librosa.feature.melspectrogram(y=sig, sr=resample_rate, n_fft=input_nfft, hop_length=input_stride)
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=resample_rate, hop_length=input_stride, y_axis='mel', x_axis='time')
    plt.show()
    plt.savefig(f'spec_{resample_rate}_{second}/{file_name}_a.png')
    plt.plot()
    plt.close()
    if idx == 4:
        break
