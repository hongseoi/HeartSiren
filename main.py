import librosa.display

import matplotlib.pyplot as plt
import numpy as np

from HeartSiren import dataset_lhg as d
import wav_to_image as wti

data_folder_path = 'data_1107/'
dataset = d.SoundData(data_folder_path)
resample_rate = 8000
dataset.resample_rate = resample_rate


for data in dataset:
    file_name = data['file_name']
    sig = data['waveform']

    # wti.plot_waveform(sig, resample_rate, file_name=file_name)

    fft = np.fft.fft(sig)
    # 복소공간 값 절댓갑 취해서, magnitude 구하기
    magnitude = np.abs(fft)

    # Frequency 값 만들기
    f = np.linspace(0, resample_rate, len(magnitude))
    left_spectrum = magnitude[:int(len(magnitude) / 2)]
    left_f = f[:int(len(magnitude) / 2)]

    # plt.figure(figsize=(10, 5))
    plt.plot(left_f, left_spectrum)
    # plt.xlabel("Frequency")
    # plt.ylabel("Magnitude")
    # plt.title("Power spectrum")
    # plt.show()
    plt.savefig(f'spec_8000_10/{file_name}_A.png', bbox_inches="tight", pad_inches=0)

    # STFT -> spectrogram
    hop_length = 512  # 전체 frame 수
    n_fft = 2048  # frame 하나당 sample 수

    # calculate duration hop length and window in seconds
    hop_length_duration = float(hop_length) / resample_rate
    n_fft_duration = float(n_fft) / resample_rate

    # STFT
    stft = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)
    # 복소공간 값 절댓값 취하기
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude)

    # display spectrogram
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(log_spectrogram, sr=resample_rate, hop_length=hop_length)
    plt.savefig(f'spec_8000_10/{file_name}.png', bbox_inches="tight", pad_inches=0)
    # plt.savefig(f'{file_name}_specgram.png')
    plt.plot()
    plt.close()
    break