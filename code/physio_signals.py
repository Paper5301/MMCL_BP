import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io
from scipy import signal
import pickle

path = 'Data/Subject17/Subject17_T1.mat'

data = scipy.io.loadmat(path)['CEAL_Data']


def filter_signal(x,fs = 1000, fc1 = 20, fc2 = 0.5, normalize = False):
    b1, a1 = signal.butter(6, fc1 / (fs / 2), btype='low')
    b2 = signal.firwin(5001, fc2 / (fs / 2), pass_zero=False)
    x_f = signal.filtfilt(b1, a1, x)
    if normalize:
        x_f = x_f - np.mean(x_f)
    x_ff = signal.filtfilt(b2, [1.0], x_f)
    return x_ff
# 1 -> BCG
# 2 -> FECG
# 3 -> FEMG L
# 4 -> EMG R
# 5 -> FPPG
# 6 -> BP
# 10 -> Ear PPG
# 11 -> Chest ECG



fs = 1000
fc1 = 20
fc2 = 0.5

#Design the Butterworth filter
b1, a1 = signal.butter(6, fc1 / (fs / 2), btype='low')

# Design the FIR filter
b2 = signal.firwin(5001, fc2 / (fs / 2), pass_zero=False)

s = np.arange(0, data.shape[1])

# ECG
CECG_raw = data[11, s]
CECG_ff = filter_signal(CECG_raw,normalize=True)

# FPPG

FPPG_raw = data[5, s]
FPPG_ff = filter_signal(FPPG_raw,normalize=True)

# BP

BP_raw = data[6, s]
BP_ff = filter_signal(BP_raw,normalize=True)

physio = {'bp':BP_ff,'ppg':FPPG_ff, 'ecg':CECG_ff}

filename = 'physio.pkl'

# Open the file in binary read mode and load the list using pickle.load()
with open(filename, 'wb') as file:
    pickle.dump(physio, file)

print(f"List saved to {filename}")