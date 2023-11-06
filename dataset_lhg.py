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

        self.max_second = m_s

    def __len__(self):
        return len(self.wav_files)

    def band_pass_filter(self, waveform):
        # filter 적용
        b, a = signal.butter(4, self.freq_high, btype='low', fs=self.sample_rate)

        num_channels = waveform.shape[0]
        for i in range(num_channels):
            waveform[i] = torch.tensor(signal.lfilter(b, a, waveform[i].numpy()))
        return waveform

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
        repeated_end = 0
        while repeated_end < self.max_second:
            for start, end, label in annotation:
                repeated_start = start + add_start
                repeated_end = end + add_start

                repeated_annotation.append((repeated_start, repeated_end, label))

                if self.max_second <= repeated_end:
                    repeated_annotation[-1] = (repeated_start, self.max_second, label)
                    break

            add_start += repeated_end

        return repeated_annotation

    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        wav_path = os.path.join(self.folder_path, wav_file)

        # Load WAV file
        waveform, sample_rate = torchaudio.load(wav_path)

        # Preprocess and repeat waveform
        processed_waveform = self.band_pass_filter(waveform)
        processed_waveform = self.repeat_waveform(processed_waveform, target_duration=self.max_second)

        # Get the corresponding TSV file
        tsv_file = os.path.splitext(wav_file)[0] + ".tsv"
        tsv_path = os.path.join(self.folder_path, tsv_file)

        # Load  the original annotation
        labels_df = pd.read_csv(tsv_path, delimiter='\t', header=None)
        original_annotation = list(zip(labels_df[0], labels_df[1], labels_df[2]))

        # Repeat and adjust the annotation
        repeated_annotation = self.repeat_annotation(original_annotation)

        # Create a new DataFrame with the repeated annotation
        repeated_labels_df = pd.DataFrame(repeated_annotation, columns=['start', 'end', 'anno'])

        return {
            "waveform": processed_waveform,
            "sample_rate": sample_rate,
            "anno_data": repeated_labels_df  # Use the repeated annotation
        }


if __name__ == '__main__':
    data_folder_path = "../data/training_data/"
    dataset = SoundData(data_folder_path)

    # Access individual data samples
    sample = dataset[3]
    print(sample)