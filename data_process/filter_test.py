
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import librosa
import librosa.display

import IPython.display as ipd

filename = "data/2530_AV.wav"

data, sr = librosa.load(filename, sr=4000)
    
def normalization(sig):
    return (sig-sig.mean())/sig.std()

print(normalization(data))

#waveform, sample_rate = torchaudio.load(filename)
#lowpass_waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq=500)
#print("Min. of lowpass_waveform: {}\nMax. of lowpass_waveform: {}\nMean of lowpass_waveform: {}".format(lowpass_waveform.min(), lowpass_waveform.max(), lowpass_waveform.mean()))

#gain_waveform = torchaudio.functional.gain(waveform, gain_db=5.0)
#print("Min. of gain_waveform {} \nMax. of gain_waveform {} \nMean of gain_waveform {}".format(gain_waveform.min(), gain_waveform.max(), gain_waveform.mean()))
#print(gain_waveform)