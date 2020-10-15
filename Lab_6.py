import numpy as np
import matplotlib.pyplot as plt
import wfdb
import pandas as pd
from scipy.signal import butter, lfilter, filtfilt

# https://physionet.org/content/wrist/1.0.0/


class PPG:

    def __init__(self, physionet_file="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/s3_walk"):

        data = wfdb.rdrecord(physionet_file)
        self.df = pd.DataFrame(data.p_signal, columns=data.sig_name)

    def filter_signal(self, data, filter_type, low_f=None, high_f=None, sample_f=256, filter_order=2):
        """Function that creates bandpass filter to ECG data.

        Required arguments:
        -data: 3-column array with each column containing one accelerometer axis
        -type: "lowpass", "highpass" or "bandpass"
        -low_f, high_f: filter cut-offs, Hz
        -sample_f: sampling frequency, Hz
        -filter_order: order of filter; integer
        """

        nyquist_freq = 0.5 * sample_f

        if filter_type == "lowpass":
            low = low_f / nyquist_freq
            b, a = butter(N=filter_order, Wn=low, btype="lowpass")
            filtered_data = lfilter(b, a, data)
            #filtered_data = filtfilt(b, a, x=data)

        if filter_type == "highpass":
            high = high_f / nyquist_freq

            b, a = butter(N=filter_order, Wn=high, btype="highpass")
            filtered_data = lfilter(b, a, data)
            # filtered_data = filtfilt(b, a, x=data)

        if filter_type == "bandpass":
            low = low_f / nyquist_freq
            high = high_f / nyquist_freq

            b, a = butter(N=filter_order, Wn=[low, high], btype="bandpass")
            filtered_data = lfilter(b, a, data)
            # filtered_data = filtfilt(b, a, x=data)

        self.df["wrist_ppg_filt"] = filtered_data

    def plot_ppg(self):
        fig, (ax1, ax2) = plt.subplots(2, sharex='col')
        ax1.plot(np.arange(0, self.df.shape[0], 1) / 256, self.df["wrist_ppg"], label="ppg", color='black')
        ax1.legend()
        ax2.plot(np.arange(0, self.df.shape[0], 1) / 256, self.df["wrist_ppg_filt"], label="ppg_filt", color='red')
        ax2.legend()


ppg = PPG()
ppg.filter_signal(data=ppg.df["wrist_ppg"], filter_type="bandpass", low_f=.1, high_f=5, sample_f=256, filter_order=2)
ppg.plot_ppg()

# Leave gyro data
# Delete low-noise accel data

# Compare gyro to accel
# Compare ECG to PPG
# Cropping: start/stop in seconds
# Data filter for PPG, default for ECG
# Peak detection for ECG and PPG: Pan-Tompkins for ECG, ? for PPG
# Time series plot of HR for PPG and ECG
# FilterDemo.py: additive sine waves + filtering + FFT
# Play around with STFT to compare PPG and ECG signal
