import os
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
import scipy.signal as signal
import torch
import torch.nn.functional as F

class SoundData(Dataset):
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.wav_files = [f for f in os.listdir(root_folder) if f.endswith(".wav")]

        # filter 정의
        self.freq_low = 0.5
        self.freq_high = 500
        self.sample_rate = 22500
        self.b, self.a = signal.butter(4, [self.freq_low, self.freq_high], btype='band', fs=self.sample_rate)

    def __len__(self):
        return len(self.wav_files)

    # 전처리 함수
    def preprocess_data(self, waveform):
      # filter 적용
        num_channels = waveform.shape[0]
        for i in range(num_channels):
            waveform[i] = torch.tensor(signal.lfilter(self.b, self.a, waveform[i].numpy()))
        return waveform

        # othertransform
        #processed_waveform = self.transforms(waveform)
        #return processed_waveform

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


    def repeat_annotation(self, annotation, repetitions):
      repeated_annotation = []
      for start, end, label in annotation:
       for i in range(repetitions):
            repeated_start = start + i * (end - start)
            repeated_end = repeated_start + (end - start)
            repeated_annotation.append((repeated_start, repeated_end, label))
      return repeated_annotation

    def __getitem__(self, idx):
      wav_file = self.wav_files[idx]
      wav_path = os.path.join(self.root_folder, wav_file)

      # Load WAV file
      waveform, sample_rate = torchaudio.load(wav_path)

      # Preprocess and repeat waveform
      processed_waveform = self.preprocess_data(waveform)
      processed_waveform = self.repeat_waveform(processed_waveform, target_duration=10.0)


      # Get the corresponding TSV file
      tsv_file = os.path.splitext(wav_file)[0] + ".tsv"
      tsv_path = os.path.join(self.root_folder, tsv_file)

      # Load  the original annotation
      labels_df = pd.read_csv(tsv_path, delimiter='\t', header=None)
      original_annotation = list(zip(labels_df[0], labels_df[1], labels_df[2]))

      # Calculate the number of repetitions based on the original annotation
      annotation_duration = labels_df[1].max() - labels_df[0].min()
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


# Example usage
root_folder = "/content/drive/MyDrive/audio prac/the-circor-digiscope-phonocardiogram-dataset-1.0.3/cut_data/"  # 데이터가 있는 폴더 경로
dataset = SoundData(root_folder)

# Access individual data samples
sample = dataset[3]
print(sample)