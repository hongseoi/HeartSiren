import os
import pandas as pd
import numpy as np

import torchaudio
from torch.utils.data import Dataset
#from torchaudio.transforms import LowPassFilter
import scipy.signal as signal
import torch
import librosa


class SoundData(Dataset):
    def __init__(self, f_p, c_o=500, m_s=10, re_rate=8000):
        """

        :param f_p: data_folder_path
        :param c_o: cut_off
        :param m_s: max_second
        :param re_rate: resample_rate
        """

        self.folder_path = f_p
        self.wav_files = [f for f in os.listdir(self.folder_path) if f.endswith(".wav")]

        self.cut_off = c_o
        self.sample_rate = 0
        self.max_second = m_s
        self.resample_rate = re_rate

    def __len__(self):
        return len(self.wav_files)

    def resampling(self, sig):
        if self.sample_rate != self.resample_rate:
            self.sample_rate = self.resample_rate
            return librosa.resample(sig, orig_sr=self.sample_rate, target_sr=self.resample_rate)

        return sig
    
    def normalization(self, sig):
        return (sig-sig.mean())/sig.std()


    def low_pass_filter(self, waveform):
        # nyq = 0.5 * self.sample_rate
        # cut_off = self.cut_off / nyq
        # b, a = signal.butter(2, cut_off, btype='low', fs=self.sample_rate)
        # return signal.filtfilt(b, a, waveform)

        waveform = torch.from_numpy(waveform)
        lowpass_waveform = torchaudio.functional.lowpass_biquad(waveform, self.sample_rate, cutoff_freq=self.cut_off, Q=1)
        lowpass_waveform = lowpass_waveform.numpy()
        return lowpass_waveform

    def repeat_waveform(self, waveform, target_duration):
        num_frames = waveform.shape[0]
        target_num_frames = int(target_duration * self.sample_rate)

        if num_frames < target_num_frames:
            # Repeat the waveform to reach the target duration
            repetitions = (target_num_frames // num_frames) + 1
            waveform = np.repeat(waveform, repetitions)
            num_frames = waveform.shape[0]

        # Trim or repeat to match the exact target duration
        if num_frames > target_num_frames:
            waveform = waveform[:target_num_frames]

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
        file_name = os.path.splitext(wav_file)[0]

        # Load WAV file
        waveform, self.sample_rate = torchaudio.load(wav_path)
        sig, sr = librosa.load(wav_path, sr=self.sample_rate)

        #sig = self.normalization(sig)
        sig = self.resampling(sig)

        # Preprocess and repeat waveform
        sig = self.low_pass_filter(sig)
        processed_waveform = self.repeat_waveform(sig, target_duration=self.max_second)

        # Get the corresponding TSV file
        tsv_file = file_name + ".tsv"
        tsv_path = os.path.join(self.folder_path, tsv_file)

        # Load  the original annotation
        labels_df = pd.read_csv(tsv_path, delimiter='\t', header=None)
        original_annotation = list(zip(labels_df[0], labels_df[1], labels_df[2]))

        # Repeat and adjust the annotation
        repeated_annotation = self.repeat_annotation(original_annotation)

        # Create a new DataFrame with the repeated annotation
        repeated_labels_df = pd.DataFrame(repeated_annotation, columns=['start', 'end', 'anno'])

        return {
            'file_name': file_name,
            "waveform": processed_waveform,
            "anno_data": repeated_labels_df  # Use the repeated annotation
        }


if __name__ == '__main__':
    data_folder_path = "./data/"
    dataset = SoundData(data_folder_path)

    # Access individual data samples
    sample = dataset[3]
    print(sample['waveform'])