import numpy as np
import matplotlib.pyplot as plt
import wfdb
import pandas as pd
from scipy.signal import butter, lfilter, filtfilt
from datetime import timedelta
import os
import peakutils
import pywt
import scipy.fft
import heartpy


# https://physionet.org/content/wrist/1.0.0/


class Data:

    def __init__(self, physionet_file="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/s3_walk",
                 fig_width=10, fig_height=6):

        self.physionet_file = physionet_file
        self.fig_width = fig_width
        self.fig_height = fig_height

        self.filter_dict = {"PPG_low": 0, "PPG_high": 0,
                            "ECG_low": 0, "ECG_high": 0,
                            "Gyro_low": 0, "Gyro_high": 0,
                            "Accel_low": 0, "Accel_high": 0}

        self.df = None
        self.ecg_peaks = []
        self.ppg_peaks = []
        self.gyro_peaks = []
        self.accel_peaks = []

        self.start_stamp = None
        self.stop_stamp = None

        self.import_data()

    def import_data(self):
        """Imports data from physionet database."""

        data = wfdb.rdrecord(self.physionet_file)
        self.df = pd.DataFrame(data.p_signal, columns=data.sig_name)
        self.df.insert(0, "Timestamp", [i / 256 for i in range(self.df.shape[0])])

        self.df = self.df[["Timestamp", "chest_ecg", "wrist_ppg", "wrist_gyro_x", "wrist_gyro_y", "wrist_gyro_z",
                           "wrist_wide_range_accelerometer_x", "wrist_wide_range_accelerometer_y",
                           "wrist_wide_range_accelerometer_z"]]
        self.df.columns = ["Timestamp", "ECG", "PPG", "Gyro_X", "Gyro_Y", "Gyro_Z",
                           "Accel_X", "Accel_Y", "Accel_Z"]

    @staticmethod
    def check_file_overwrite(filename):

        existing_files = os.listdir(os.getcwd())

        pngs = [i for i in existing_files if "png" in i]

        near_matches = []

        if len(pngs) > 0:

            for png in pngs:
                if filename not in png:
                    file_exists = False
                if filename in png:
                    file_exists = True
                    near_matches.append(png)

        if len(pngs) == 0:
            file_exists = False

        if file_exists:
            near_matches = sorted(near_matches)
            print(near_matches)
            versions = []

            for near_match in near_matches:
                if "Version" in near_match:
                    versions.append(int(near_match.split("Version")[1].split(".")[0]))
                if "Version" not in near_match:
                    pass

            if len(versions) >= 1:
                version = max(versions) + 1
            if len(versions) == 0:
                version = 1

            print("File already exists.")
            print("Saving new file as {}".format(filename.split(".")[0] + "_Version{}.png".format(version)))

            return filename.split(".")[0] + "_Version{}.png".format(version)

        if not file_exists:
            return filename

    def get_timestamps(self, start=None, stop=None):
        """Crops data using start/stop arguments (number of seconds into file)"""

        start_stamp = self.df["Timestamp"].iloc[0] + timedelta(seconds=start)
        stop_stamp = self.df["Timestamp"].iloc[0] + timedelta(seconds=stop)

    def filter_signal(self, data, filter_type, low_f=None, high_f=None, sample_f=256, filter_order=2, data_type="PPG"):
        """Function that creates bandpass filter to ECG data.

        Required arguments:
        -data: 3-column array with each column containing one accelerometer axis
        -type: "lowpass", "highpass" or "bandpass"
        -low_f, high_f: filter cut-offs, Hz
        -sample_f: sampling frequency, Hz
        -filter_order: order of filter; integer
        """

        if data_type == "PPG":
            self.filter_dict["PPG_low"] = low_f
            self.filter_dict["PPG_high"] = high_f
        if data_type == "ECG":
            self.filter_dict["ECG_low"] = low_f
            self.filter_dict["ECG_high"] = high_f
        if data_type == "Gyro":
            self.filter_dict["Gyro_low"] = low_f
            self.filter_dict["Gyro_high"] = high_f
        if data_type == "Accel":
            self.filter_dict["Accel_low"] = low_f
            self.filter_dict["Accel_high"] = high_f

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

        return filtered_data

    def plot_ppg_filter(self):
        """Plots raw and filtered PPG data."""

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(self.fig_width, self.fig_height))

        ax1.set_title("PPG Data: Raw vs. {}-{} Hz bandpass".format(float(self.filter_dict["PPG_low"]),
                                                                   float(self.filter_dict["PPG_high"])))

        ax1.plot(np.arange(0, self.df.shape[0], 1) / 256, self.df["PPG"], label="PPG_raw", color='black')
        ax1.legend()
        ax2.plot(np.arange(0, self.df.shape[0], 1) / 256, self.df["PPG_Filt"], label="PPG_filt", color='red')
        ax2.plot([i / 256 for i in self.ppg_peaks], [self.df.iloc[i]["PPG_Filt"] for i in self.ppg_peaks],
                 linestyle="", marker="o", color='black', markersize=4)
        ax2.legend()

    def plot_ecg_filter(self):
        """Plots raw and filtered ECG data. Plots peaks if peak detection has been run."""

        if self.start_stamp is not None and self.stop_stamp is not None:
            df = self.df.loc[(self.df["Timestamp"] >= self.start_stamp) & (self.df["Timestamp"] <= self.stop_stamp)]
        if self.start_stamp is None and self.stop_stamp is None:
            df = self.df

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(self.fig_width, self.fig_height))

        ax1.set_title("ECG Data: Raw vs. {}-{} Hz bandpass".format(float(self.filter_dict["ECG_low"]),
                                                                   float(self.filter_dict["ECG_high"])))

        ax1.plot(np.arange(0, df.shape[0], 1) / 256, df["ECG"], label="ECG_raw", color='black')
        ax1.legend()

        ax2.plot(np.arange(0 / 256, df.shape[0], 1) / 256, df["ECG_Filt"], label="ECG_filt", color='red')

        ax2.plot([i / 256 for i in self.ecg_peaks], [df.iloc[i]["ECG_Filt"] for i in self.ecg_peaks],
                 linestyle="", marker="o", color='black', markersize=4)

        ax2.legend()

    def detect_ecg_peaks(self, start=None, stop=None, pantompkins=True, thresh=.5, abs_thresh=False):
        """Pan-Tompkins peak detection algorithm (1985).

        Code taken from ecgdetectors package (Luis Howell & Bernd Porr): https://github.com/luishowell/ecg-detectors

        :argument
        -start/stop: data cropping, seconds into collection
        -pantompkins: boolean to run stationary wavelet + Pan-Tompkins algorith. Runs peakutils if False
        -thresh: threshold used if pantompkins = False
        -abs_thresh: boolean to use absolute vs. normalized threshold for peakutils peak detection
        """

        if start is not None and stop is not None:
            df = self.df.loc[(self.df["Timestamp"] >= start) & (self.df["Timestamp"] <= stop)]
        if start is None and stop is None:
            df = self.df

        self.start_stamp = start
        self.stop_stamp = stop

        if pantompkins:

            unfiltered_ecg = [i for i in df["ECG"]]

            swt_level = 3
            padding = -1
            for i in range(1000):
                if (len(unfiltered_ecg) + i) % 2 ** swt_level == 0:
                    padding = i
                    break

            if padding > 0:
                unfiltered_ecg = np.pad(unfiltered_ecg, (0, padding), 'edge')
            elif padding == -1:
                print("Padding greater than 1000 required\n")

            swt_ecg = pywt.swt(unfiltered_ecg, 'db3', level=swt_level)
            swt_ecg = np.array(swt_ecg)
            swt_ecg = swt_ecg[0, 1, :]

            squared = swt_ecg * swt_ecg

            f1 = 0.01 / 256
            f2 = 10 / 256

            b, a = butter(3, [f1 * 2, f2 * 2], btype='bandpass')
            filtered_squared = lfilter(b, a, squared)

            def panPeakDetect(detection, fs):
                """Detection = filtered + squared data"""

                # Corresponds to HR of 240
                min_distance = int(0.25 * fs)

                signal_peaks = [0]
                noise_peaks = []

                SPKI = 0.0  # "running estimate of signal peak"
                NPKI = 0.0  # "running estimate of noise peak", includes non-QRS ECG waves

                threshold_I1 = 0.0  # first threshold
                threshold_I2 = 0.0  # second threshold; used only if necessary (used if no QRS found in specified region)

                RR_missed = 0  # estimate of how many potential heartbeats were missed in a period with no found peaks..?
                index = 0
                indexes = []

                missed_peaks = []
                peaks = []

                # Loops through input data --------------------------------------------------------------------------------
                for i in range(len(detection)):

                    # Continues loop while index in data
                    if i > 0 and i < len(detection) - 1:

                        # Looks for inflection point in data; adds index to "peaks"
                        if detection[i - 1] < detection[i] and detection[i + 1] < detection[i]:
                            peak = i
                            peaks.append(i)

                            # Requires datapoint at peak to be above threshold AND more than 300 ms after previous peak
                            # (likely to be true QRS
                            if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * fs:

                                # Adds index of QRS to signal_peaks
                                signal_peaks.append(peak)
                                indexes.append(index)

                                # Initial threshold setting
                                SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI

                                if RR_missed != 0:

                                    if signal_peaks[-1] - signal_peaks[-2] > RR_missed:

                                        # Sets indexes for region with no detected peaks
                                        missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]  # previous 8 beats
                                        missed_section_peaks2 = []  # previous 8 beats within RR interval limits

                                        # Loops through section of no detected peaks
                                        for missed_peak in missed_section_peaks:

                                            # Requires potential peaks to be adequately spaced AND
                                            # data value above threshold_I2
                                            if missed_peak - signal_peaks[-2] > min_distance and \
                                                    signal_peaks[-1] - missed_peak > min_distance and \
                                                    detection[missed_peak] > threshold_I2:
                                                missed_section_peaks2.append(missed_peak)

                                        if len(missed_section_peaks2) > 0:
                                            missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                                            missed_peaks.append(missed_peak)
                                            signal_peaks.append(signal_peaks[-1])
                                            signal_peaks[-2] = missed_peak

                            else:
                                noise_peaks.append(peak)
                                NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI  # adjusts noise threshold

                            threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)  # adjusts signal threshold
                            threshold_I2 = 0.5 * threshold_I1  # adjusts secondary signal threhsold

                            if len(signal_peaks) > 8:
                                RR = np.diff(signal_peaks[-9:])  # RR intervals for previous 8 beats
                                RR_ave = int(np.mean(RR))  # average RR interval over last 8 beats
                                RR_missed = int(1.66 * RR_ave)  #

                            index = index + 1

                signal_peaks.pop(0)

                return signal_peaks

            filt_peaks = panPeakDetect(detection=np.asarray([i for i in filtered_squared]), fs=256)

        if not pantompkins:

            filt_peaks = peakutils.indexes(y=self.df["ECG_Filt"], thres=thresh, thres_abs=abs_thresh, min_dist=80)

        self.ecg_peaks = filt_peaks

        average_hr = (len(filt_peaks) - 1) / (self.df.shape[0] / 256) * 60
        print("-Average HR is {} bpm.".format(round(average_hr, 1)))

    def detect_ppg_peaks(self):

        pass

    def plot_ecg_ppg_fft(self):

        ppg = self.filter_signal(data=x.df["PPG"].dropna(), filter_type="highpass", high_f=.05, sample_f=256)
        ecg = self.filter_signal(data=x.df["ECG"].dropna(), filter_type="highpass", high_f=.05, sample_f=256)

        fft_ppg = scipy.fft.fft(ppg)
        fft_ecg = scipy.fft.fft(ecg)

        fs = 256
        l = x.df.shape[0]

        xf = np.linspace(0.0, 1.0 / (2.0 * (1 / fs)), l // 2)

        ppg_power = 2.0 / l / 2 * np.abs(fft_ppg[0:l // 2])
        ecg_power = 2.0 / l / 2 * np.abs(fft_ecg[0:l // 2])

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(self.fig_width, self.fig_height))
        plt.suptitle("Fast Fourier Transform Comparison (.05Hz highpass filter to remove DC)")

        ax1.plot(xf, ppg_power, color='dodgerblue', label="PPG")
        ax1.set_ylabel("Power")
        ax1.legend()

        ax2.plot(xf, ecg_power, color='red', label="ECG")
        ax2.set_ylabel("Power")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.legend()

        ax2.set_xlim(-1, 30)

    def demo_heartpy(self):

        pass


x = Data()

"""
x.df["PPG_Filt"] = x.filter_signal(data=x.df["PPG"], filter_type="bandpass", low_f=.67, high_f=3, sample_f=256)
x.detect_ppg_peaks()
x.plot_ppg_filter()

x.df["ECG_Filt"] = x.filter_signal(data=x.df["ECG"], filter_type="bandpass", low_f=.67, high_f=25, data_type="ECG")
x.detect_ecg_peaks(pantompkins=True, start=None, stop=None)
x.plot_ecg_filter()
"""

# WILL NEED PYWT PACKAGE ON JUPYTER


def heartpy_demo():

    ecg_data = np.asarray(x.df["ECG"])
    ecg_filt = x.filter_signal(data=ecg_data[:-500], data_type="ECG", filter_type='bandpass', low_f=1, high_f=15)
    wd_ecg, m_ecg = heartpy.process(heartpy.scale_data(ecg_filt), 256)

    ppg_data = np.asarray(x.df["PPG"])
    ppg_filt = x.filter_signal(data=ppg_data[:-500], data_type="PPG", filter_type='bandpass', low_f=.4, high_f=3)
    wd_ppg, m_ppg = heartpy.process(heartpy.scale_data(ppg_filt), 256)

    plt.plot(wd_ecg['hr'], color='red')
    plt.plot(wd_ppg['hr'], color='black')

    # PLOTTING ------------------------------------------------------------------------------------------------------
    """fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 6))

    # ECG with peaks
    ax1.plot([i/256 for i in np.arange(len(wd_ecg['hr']))], wd_ecg['hr'], color='black', label="ECG")
    ax1.plot([np.arange(len(wd_ecg["hr"]))[i]/256 for i in wd_ecg["peaklist_cor"]],
             [wd_ecg['hr'][i] for i in wd_ecg['peaklist_cor']], linestyle="", color='red', marker='o')
    ax1.set_ylabel("Voltage")
    ax1.legend()

    # PPG with peaks
    ax2.plot([i/256 for i in np.arange(len(wd_ppg['hr']))], wd_ppg['hr'], color='black', label="PPG")
    ax2.plot([np.arange(len(wd_ppg["hr"]))[i]/256 for i in wd_ppg["peaklist_cor"]],
             [wd_ppg['hr'][i] for i in wd_ppg['peaklist_cor']], linestyle="", color='dodgerblue', marker='o')
    ax2.set_ylabel("PPG raw unit?")
    ax2.legend()

    # beat to beat HR
    ax3.plot([i/256 for i in wd_ecg['peaklist']], [1000/i*60 for i in wd_ecg['ybeat']], color='red', label="ECG")
    ax3.plot([i/256 for i in wd_ppg['peaklist']], [1000/i*60 for i in wd_ppg['ybeat']], color='black', label='PPG')
    ax3.legend()
    ax3.set_ylabel("HR (bpm)")
    ax3.set_xlabel("Seconds")"""

    return wd_ecg, wd_ppg


ecg, ppg = heartpy_demo()
