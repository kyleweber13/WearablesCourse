import numpy as np
import matplotlib.pyplot as plt
import wfdb
import pandas as pd

# https://physionet.org/content/wrist/1.0.0/

x = wfdb.rdrecord("/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/s3_walk")
df = pd.DataFrame(x.p_signal, columns=x.sig_name)

fig, (ax1, ax2) = plt.subplots(2, sharex='col')
ax1.plot(np.arange(0, df.shape[0], 1) / 256, df["wrist_ppg"], label="ppg", color='black')
ax1.legend()
ax2.plot(np.arange(0, df.shape[0], 1) / 256, df["chest_ecg"], label="ecg", color='red')
ax2.legend()

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
