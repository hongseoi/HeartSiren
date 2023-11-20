import librosa.display

import matplotlib.pyplot as plt
import numpy as np

import dataset as d
import wav_to_image as wti
import os

# torchaudio
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

data_folder_path = '../../data_1107/'

dataset = d.SoundData(data_folder_path)

resample_rate = 8000
dataset.resample_rate = resample_rate

second = 5
dataset.max_second = second

output_folder = f'spec_{resample_rate}_{second}'
dataset.output_folder = output_folder

for idx, data in enumerate(dataset):
    file_name = data['file_name']
    sig = data['waveform']

    # STFT -> spectrogram
    #hop_length = 512
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
    print("last shape:", sig.shape)
    print('mel:', librosa.power_to_db(mel))
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=resample_rate, hop_length=input_stride) #, y_axis='mel', x_axis='time'
    plt.savefig(f'{output_folder}/{file_name}.png', bbox_inches="tight", pad_inches=0)
    plt.plot()
    plt.close()
    # if idx == 20:
    #     break

# # 폴더가 없으면 생성
# output_folder = "./img/"
# os.makedirs(output_folder, exist_ok=True)

# # 변환 함수 리스트
# mel_transforms = [
#     T.MelSpectrogram(sample_rate=resample_rate, n_fft=300, hop_length=800, n_mels=64),
#     T.MelSpectrogram(sample_rate=resample_rate, n_fft=400, hop_length=800, n_mels=64),
#     T.MelSpectrogram(sample_rate=resample_rate, n_fft=500, hop_length=800, n_mels=64),
#     T.MelSpectrogram(sample_rate=resample_rate, n_fft=600, hop_length=800, n_mels=64),
#     T.MelSpectrogram(sample_rate=resample_rate, n_fft=700, hop_length=800, n_mels=64),
#     T.MelSpectrogram(sample_rate=resample_rate, n_fft=800, hop_length=800, n_mels=64),
#     T.MelSpectrogram(sample_rate=resample_rate, n_fft=900, hop_length=800, n_mels=64),


#     #T.MelSpectrogram(sample_rate=resample_rate, n_fft=400, hop_length=512, n_mels=128),
#     #T.MelSpectrogram(sample_rate=resample_rate, n_fft=400, hop_length=256, n_mels=64),
#     #T.MelSpectrogram(sample_rate=resample_rate, n_fft=400, hop_length=256, n_mels=128), # n_fft 400 512 1024
#     # 다른 매개변수 추가
# ]

# # 데이터셋 순회
# for idx, data in enumerate(dataset):
#     file_name = data['file_name']
#     sig = data['waveform']

#     # 각 변환 함수에 대해 스펙트로그램 생성 및 저장
#     for i, mel_transform in enumerate(mel_transforms):

#         mel = mel_transform(torch.tensor(sig).unsqueeze(0)) #배치차원추가
#         mel_db = T.AmplitudeToDB()(mel)

#         # 스펙트로그램 저장
#         plt.figure(figsize=(10, 5))
#         plt.imshow(mel_db[0].numpy(), cmap='viridis', aspect='auto', origin='lower')
#         plt.ylim(0, mel_db.shape[1])  
#         plt.xlim(0, mel_db.shape[2]) 
#         plt.savefig(f'{output_folder}/{file_name}_transform_{i}.png', bbox_inches="tight", pad_inches=0)
#         plt.close()

#     if idx == 5:
#         break