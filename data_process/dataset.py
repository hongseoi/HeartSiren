import os
import pandas as pd
import numpy as np

import torchaudio
from torch.utils.data import Dataset
#from torchaudio.transforms import LowPassFilter
import scipy.signal as signal
import torch
import librosa
from biquad import Biquad

import multiprocessing


class SoundData(Dataset):
    def __init__(self, f_p, c_o=500, m_s=10, re_rate=8000, o_p="./processed_data", filter_repetition_count=1):

        """

        :param f_p: data_folder_path
        :param c_o: cut_off
        :param m_s: max_second
        :param re_rate: resample_rate
        """

        self.folder_path = f_p
        self.output_folder = o_p

        # 폴더가 존재하지 않으면 생성
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        self.wav_files = [f for f in os.listdir(self.folder_path) if f.endswith(".wav")]
        self.high_cut = c_o
    
        self.sample_rate = None
        self.max_second = m_s
        self.resample_rate = re_rate

        self.filter_repetition_count = filter_repetition_count

    def __len__(self):
        return len(self.wav_files)

    def resampling(self, sig):
        if self.sample_rate != self.resample_rate:
            sig = librosa.resample(sig, orig_sr=self.sample_rate, target_sr=self.resample_rate)
            self.sample_rate = self.resample_rate
            return sig            

        return sig

    def normalization(self, sig):
        return (sig-sig.mean())/sig.std()

    def low_pass_filter(self, wf):
        # nyq = 0.5 * self.sample_rate
        # cut_off = self.cut_off / nyq
        # b, a = signal.butter(2, cut_off, btype='low', fs=self.sample_rate)
        # return signal.filtfilt(b, a, waveform)

        waveform = torch.from_numpy(wf)
        for i in range(self.filter_repetition_count):
            waveform = torchaudio.functional.lowpass_biquad(waveform, self.sample_rate, cutoff_freq=self.high_cut, Q=0.707)
        waveform = waveform.numpy()
        return waveform
        
        # wf = torch.from_numpy(wf)
        # lowpass_cutoff = 500
        # bq_lowquid = Biquad(Biquad.LOWPASS, lowpass_cutoff, self.sample_rate)
        # wf = torchaudio.functional.biquad(wf, bq_lowquid.b0, bq_lowquid.b1, bq_lowquid.b2, 1., bq_lowquid.a1, bq_lowquid.a2)
        # print(type(wf))
        # return  wf.numpy()
    
    # def high_pass_filter(self, waveform):
    #     waveform = torch.from_numpy(waveform)
    #     highpass_waveform = torchaudio.functional.highpass_biquad(waveform, self.sample_rate, cutoff_freq=self.high_cut, Q=1, reset_state=True)
    #     return highpass_waveform.numpy()
    
    def band_pass_filter(self, waveform):
        waveform = torch.from_numpy(waveform)
        band_pass_waveform = torchaudio.functional.bandpass_biquad(waveform, self.sample_rate, central_freq=500, Q=2)
        return waveform.numpy()
    
    def repeat_waveform(self, waveform):
        num_frames = waveform.shape[0]

        target_num_frames = int(self.max_second * self.sample_rate)

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
                    del repeated_annotation[-1]
                    break

            add_start = repeated_end

        return repeated_annotation

    # def __getitem__(self, idx):
    # wav_file = self.wav_files[idx]
    def make_csv(self, wav_file):
        wav_path = os.path.join(self.folder_path, wav_file)
        file_name = os.path.splitext(wav_file)[0]

        # Load WAV file
        waveform, self.sample_rate = torchaudio.load(wav_path)
        sig, sr = librosa.load(wav_path, sr=self.sample_rate)

        sig = self.resampling(sig)

        sig = self.low_pass_filter(sig)

        processed_waveform = self.repeat_waveform(sig)
       
        # Get the corresponding TSV file
        tsv_file = file_name + ".tsv"
        tsv_path = os.path.join(self.folder_path, tsv_file)

        # Load  the original annotation
        labels_df = pd.read_csv(tsv_path, delimiter='\t', header=None)
        original_annotation = list(zip(labels_df[0], labels_df[1], labels_df[2]))

        # Repeat and adjust the annotation
        repeated_annotation = self.repeat_annotation(original_annotation)

        # Create a new DataFrame with the repeated annotation
        repeated_labels_df = pd.DataFrame(repeated_annotation, columns=['start', 'end', 'annotations'])

        # # Save the new TSV file in the output folder
        output_path = os.path.join(self.output_folder, file_name + '_' + str(self.filter_repetition_count) + ".tsv")
        repeated_labels_df.to_csv(output_path, sep='\t', header=False, index=False)

        # return {
        #     'file_name': file_name,
        #     "waveform": processed_waveform,
        #     'resample_rate': self.resample_rate,
        #     'second': self.max_second,
        #     'filter_repetition_count': self.filter_repetition_count,
        #     'output_folder': self.output_folder,
        #     "anno_data": repeated_labels_df  # Use the repeated annotation
        # }

        return (file_name, processed_waveform, self.resample_rate, self.max_second, self.filter_repetition_count,
                self.output_folder)


if __name__ == '__main__':
    data_folder_path = "./raw_data/"
    output_folder = "./"

    dataset = SoundData(data_folder_path, o_p=output_folder)

    # Access individual data samples
    sample = dataset[200]
    print(sample['waveform'])
    print(sample['file_name'])
    