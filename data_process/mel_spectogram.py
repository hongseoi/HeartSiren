import matplotlib.pyplot as plt
import librosa.display
from PIL import Image
import numpy as np


class MelSpecto:
    def __init__(self, s_r, o_f, f_n, n_mels=80):
        self.sample_rate = s_r
        self.output_folder = o_f
        self.file_name = f_n
        self.n_mels = n_mels

    def image_save(self, sig, figsize):
        """
        1. n_fft : length of the windowed signal after padding with zeros.

              한 번 fft를 해 줄 만큼의 sequence 길이

        2. hop_length : window 간의 거리 라이브러리 default = 512

        3. win_length : window 길이

        :param sig: wave_form
        :param m_s: max_second
        :return:
        """
        plt.figure(figsize=figsize)
        # n_fft = 512
        # hop_length = 200
        # frame_length = 0.064
        # frame_stride = 0.025
        # n_fft = int(round(self.sample_rate * frame_length))
        # hop_length = int(round(self.sample_rate * frame_stride))

        '''
        win length: 
         - 음성을 작은 조각으로 자를 때 작은 조각의 크기를 의미
         - 자연어 처리 분야에서 25ms의 크기를 기본으로 함 
         - sr / 40
         
        n_fft
         - win_length의 크기로 잘린 음성의 작은 조각은 0으로 패딩되어서 n_fft로 크기가 맞춰짐
         - 패딩된 조각에 푸리에현봔이 적용되기 때문에 n_fft는 win_length보다 크거나 같아야하고 일반적으로 속도를 위해 2의 제곱근으로 설정함
                     
        hop_length
         - 음성을 작은 조각으로 자를 때 자르는 간격을 의미
         - 이 길이 만큼 옆으로 밀면서 작은 조각을 얻음
         - 일반적으로 10ms의 크기를 기본으로 함
         
        n_mels 
         - 적용할 mel filter의 개수를 의미
         
         
         
        우리가 보유한 데이터가 7초이고, window_length를 0.025초, frame_stride를 0.010초(10ms)단위로 뽑는다고 가정하면, 
        1칸은 0.015초(15ms)가 겹치도록 하여 총 700칸을 얻을 수 있다.
        '''
        frame_length = 0.025
        frame_stride = 0.01
        win_length = int(round(self.sample_rate * frame_length))
        n_fft = win_length
        hop_length = int(round(self.sample_rate * frame_stride))

        # n_mels = ?

        # display spectrogram
        mel = librosa.feature.melspectrogram(y=np.array(sig), sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length,
                                             win_length=win_length)
        # , y_axis='mel', x_axis='time'
        librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=self.sample_rate, hop_length=hop_length)

        # display spectrogram
        # mel = librosa.feature.melspectrogram(y=np.array(sig), sr=self.sample_rate)
        # # , y_axis='mel', x_axis='time'
        # librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=self.sample_rate)

        plt.savefig(f'{self.output_folder}/{self.file_name}.png', bbox_inches="tight", pad_inches=0)
        plt.plot()
        plt.close()

        # 삭제 말구 잘리는 annotation 부분 지우기
        # img = F.to_pil_image(image)
        # image = Image.open(f'{self.output_folder}/{self.file_name}.png')
        # w, h = image.size
        # # crop을 통해 이미지 자르기       (left,up, rigth, down)
        # croppedImage = image.crop((0, 0, w - 10, h))
        # croppedImage.save(f'{self.output_folder}/{self.file_name}.png')

