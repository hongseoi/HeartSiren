import os
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
import scipy.signal as signal
import torch
import torch.nn.functional as F


class SoundData(Dataset):
    def __init__(self, f_p, f_l=0.5, f_h=500, s_r=4000, m_s=10):
        """

        :param f_p: data_folder_path
        :param f_l: freq_low
        :param f_h: freq_high
        :param s_r: sample_rate
        :param m_s: max_second
        """

        self.folder_path = f_p
        self.wav_files = [f for f in os.listdir(self.folder_path) if f.endswith(".wav")]

        # filter 정의
        self.freq_low = f_l
        self.freq_high = f_h
        self.sample_rate = s_r
        self.b, self.a = signal.butter(4, [self.freq_low, self.freq_high], btype='band', fs=self.sample_rate)

        self.max_second = m_s

    def __len__(self):
        return len(self.wav_files)

    # 전처리 함수
    def band_pass_filter(self, waveform):
        # filter 적용
        num_channels = waveform.shape[0]
        for i in range(num_channels):
            waveform[i] = torch.tensor(signal.lfilter(self.b, self.a, waveform[i].numpy()))
        return waveform

        # othertransform
        # processed_waveform = self.transforms(waveform)
        # return processed_waveform

    # repeat 기법 사용해 데이터 길이 10초로 동일화
    def repeat_waveform(self, waveform, target_duration):
        num_channels, num_frames = waveform.shape
        target_num_frames = int(target_duration * self.sample_rate)

        if num_frames < target_num_frames:
            # Repeat the waveform to reach the target duration
            repetitions = target_num_frames // num_frames
            waveform = waveform.repeat(1, repetitions)
            num_frames = waveform.shape[1]

        # Trim or repeat to match the exact target duration
        if num_frames > target_num_frames:
            waveform = waveform[:, :target_num_frames]
        elif num_frames < target_num_frames:
            padding = torch.zeros(num_channels, target_num_frames - num_frames)
            waveform = torch.cat((waveform, padding), dim=1)

        return waveform

    def repeat_annotation(self, annotation):
        repeated_annotation = []
        add_start = 0
        while True:            
            for start, end, label in annotation:
                repeated_start = start + add_start
                repeated_end = end + add_start
                if repeated_end > 10:
                    repeated_end = 10
                repeated_annotation.append((repeated_start, repeated_end, label))

        return repeated_annotation

    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        wav_path = os.path.join(self.folder_path, wav_file)

        # Load WAV file
        waveform, sample_rate = torchaudio.load(wav_path)

        # Preprocess and repeat waveform
        processed_waveform = self.band_pass_filter(waveform)
        processed_waveform = self.repeat_waveform(processed_waveform, target_duration=10.0)

        # Get the corresponding TSV file
        tsv_file = os.path.splitext(wav_file)[0] + ".tsv"
        tsv_path = os.path.join(self.folder_path, tsv_file)

        # Load  the original annotation
        labels_df = pd.read_csv(tsv_path, delimiter='\t', header=None)
        original_annotation = list(zip(labels_df[0], labels_df[1], labels_df[2]))

        # Calculate the number of repetitions based on the original annotation
        annotation_duration = labels_df[1].max() - labels_df[0].min()
        print('dura', annotation_duration, int(10/ annotation_duration))
        num_repetitions = int(10.0 / annotation_duration)

        # Repeat and adjust the annotation
        repeated_annotation = self.repeat_annotation(original_annotation, num_repetitions)

        # Create a new DataFrame with the repeated annotation
        repeated_labels_df = pd.DataFrame(repeated_annotation, columns=[0, 1, 2])

        return {
            "waveform": processed_waveform,
            "sample_rate": sample_rate,
            "labels": repeated_labels_df  # Use the repeated annotation
        }


data_folder_path = "../data/training_data/"
dataset = SoundData(data_folder_path)

# Access individual data samples
sample = dataset[3]
print(sample)