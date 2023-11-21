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
        # display spectrogram
        mel = librosa.feature.melspectrogram(y=np.array(sig), sr=self.sample_rate, n_mels=self.n_mels)
        # , y_axis='mel', x_axis='time'
        librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=self.sample_rate, fmax=8000)
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

