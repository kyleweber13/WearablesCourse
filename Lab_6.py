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

        self.ecg_dict = {}
        self.ppg_dict = {}

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

    def filter_signal(self, data, filter_type, low_f=None, high_f=None,
                      sample_f=256, filter_order=2, data_type="PPG"):
        """Function that creates filter for PPG or ECG data.

        Required arguments:
        -data: data to filter
        -type: "lowpass", "highpass" or "bandpass"
        -low_f, high_f: filter cut-offs, Hz
        -sample_f: sampling frequency, Hz
        -filter_order: order of filter; integer
        """

        if data_type == "PPG":
            self.filter_dict["PPG_low"] = low_f
            self.filter_dict["PPG_high"] = high_f
            self.filter_dict["PPG_type"] = filter_type

        if data_type == "ECG":
            self.filter_dict["ECG_low"] = low_f
            self.filter_dict["ECG_high"] = high_f
            self.filter_dict["ECGG_type"] = filter_type

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

    def filter_imu_signal(self, data_type, filter_type, low_f=None, high_f=None,
                          sample_f=256, filter_order=2):
        """Filter function for accelerometer/gyroscope data.

        Required arguments:
        -data: data to filter
        -type: "lowpass", "highpass" or "bandpass"
        -low_f, high_f: filter cut-offs, Hz
        -sample_f: sampling frequency, Hz
        -filter_order: order of filter; integer
        """

        def filter_data(data, sample_f):
            nyquist_freq = 0.5 * sample_f

            if filter_type == "lowpass":
                low = low_f / nyquist_freq
                b, a = butter(N=filter_order, Wn=low, btype="lowpass")
                filtered_data = lfilter(b, a, data)
                # filtered_data = filtfilt(b, a, x=data)

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

        if data_type == "Gyro" or data_type == "gyro" or "Gyro" in data_type.capitalize():
            self.filter_dict["Gyro_low"] = low_f
            self.filter_dict["Gyro_high"] = high_f
            self.filter_dict["Gyro_type"] = filter_type

            self.df["Gyro_X_Filt"] = filter_data(data=self.df["Gyro_X"], sample_f=sample_f)
            self.df["Gyro_Y_Filt"] = filter_data(data=self.df["Gyro_Y"], sample_f=sample_f)
            self.df["Gyro_Z_Filt"] = filter_data(data=self.df["Gyro_Z"], sample_f=sample_f)

        if data_type == "Accel" or data_type == "accel" or "Accel" in data_type.capitalize():
            self.filter_dict["Accel_low"] = low_f
            self.filter_dict["Accel_high"] = high_f
            self.filter_dict["Accel_type"] = filter_type

            self.df["Accel_X_Filt"] = filter_data(data=self.df["Accel_X"], sample_f=sample_f)
            self.df["Accel_Y_Filt"] = filter_data(data=self.df["Accel_Y"], sample_f=sample_f)
            self.df["Accel_Z_Filt"] = filter_data(data=self.df["Accel_Z"], sample_f=sample_f)

    def ecg_find_and_plot_peaks(self, start=None, stop=None, show_plot=True):
        """Plots raw and filtered ECG data with detected peaks.

            :argument
            -start/stop: able to crop data, number of seconds
            -show_plot: boolean whether to show plot
        """

        self.start_stamp = start
        self.stop_stamp = stop

        print("\nDetecting and plotting ECG peaks...")

        self.detect_ecg_peaks(start=start, stop=stop)

        if self.start_stamp is None:
            start_offset = 0
        if self.start_stamp is not None:
            start_offset = self.start_stamp

        if show_plot:
            # PLOTTING ---------
            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(self.fig_width, self.fig_height))

            ax1.set_title("ECG Data: Raw vs. {}-{} Hz bandpass filtered".format(float(self.filter_dict["ECG_low"]),
                                                                                float(self.filter_dict["ECG_high"])))

            ax1.plot(np.arange(0, len(self.ecg_dict["Data"]), 1) / 256 + start_offset, self.ecg_dict["Data"],
                     label="ECG_raw", color='black')
            ax1.set_ylabel("Voltage")
            ax1.legend()

            ax2.plot(np.arange(0 / 256, len(self.ecg_dict["Data"]), 1) / 256 + start_offset,
                     self.ecg_dict["Data_Filt"], label="ECG_filt", color='black')

            ax2.plot([i / 256 + self.start_stamp for i in self.ecg_dict["Peaks"]],
                     [self.ecg_dict["Data_Filt"][i] for i in self.ecg_dict["Peaks"]],
                     linestyle="", marker="v", color='red', markersize=4)

            ax2.set_ylabel("Voltage")
            ax2.legend()

            ax3.plot([i / 256 + self.start_stamp for i in self.ecg_dict["Peaks"][:-1]],
                     self.ecg_dict["Beat HR"], color='red', label='Beat-to-beat HR')
            ax3.legend()
            ax3.set_ylabel("HR (bpm)")
            ax3.set_xlabel("Seconds")
            ax3.set_ylim(40, 180)

    def plot_ecg_steps(self, start=None, stop=None):
        """Generates plots that shows steps involved in ECG peak detection algorithm.

            :argument
            -start/stop: crop data, number of seconds
        """

        print("\nPlotting ECG data processing steps...")

        if "ECG_Filt" not in self.df.columns:
            print("-Filtered ECG data is required. Please run filtering function and try again.")
            return None

        # Checks to make sure loaded data matches desired section
        if self.start_stamp == start and self.stop_stamp == stop:
            pass

        if self.start_stamp != start or self.stop_stamp != stop or self.start_stamp is None or self.stop_stamp is None:
            self.ecg_find_and_plot_peaks(start=start, stop=stop, show_plot=False)

        if self.start_stamp is None and self.stop_stamp is None:
            self.start_stamp = 0
            self.stop_stamp = int(self.df.shape[0] / 256)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex="col", figsize=(self.fig_width, self.fig_height))

        if self.start_stamp is not None and self.stop_stamp is not None:
            plt.suptitle("ECG Processing Steps (time: {} to {} seconds)".format(self.start_stamp, self.stop_stamp))
        if self.start_stamp is None and self.stop_stamp is None:
            plt.suptitle("ECG Processing Steps (full file)")

        # Raw ECG
        ax1.plot(np.arange(len(self.ecg_dict["Data"]))/256 + self.start_stamp, self.ecg_dict["Data"],
                 color='red', label="Raw")

        if self.start_stamp is not None:
            ax1.set_xlim(self.start_stamp - 2, self.stop_stamp + (self.stop_stamp - self.start_stamp) * 0.2)
        if self.start_stamp is None:
            ax1.set_xlim(-2, len(self.ecg_dict["Data"])/256 * 1.2)
        ax1.set_ylabel("Voltage")
        ax1.legend(loc='center right')

        # Filtered ECG
        ax2.plot(np.arange(len(self.ecg_dict["Data_Filt"]))/256 + self.start_stamp, self.ecg_dict["Data_Filt"],
                 color='blue', label="Filtered \n({}-{}Hz)".format(self.filter_dict["ECG_low"],
                                                                 self.filter_dict["ECG_high"]))
        ax2.set_ylabel("Voltage")
        ax2.legend(loc='center right')

        # Wavelet ECG
        ax3.plot(np.arange(len(self.ecg_dict["Wavelet"])) / 256 + self.start_stamp, self.ecg_dict["Wavelet"],
                 color='green', label="Wavelet")
        ax3.set_ylabel("Voltage")
        ax3.legend(loc='center right')

        # Wavelet squared + filtered
        ax4.plot(np.arange(len(self.ecg_dict["Squared"]))/256 + self.start_stamp, self.ecg_dict["Squared"],
                 color='dodgerblue', label="Squared")

        ax4.plot([np.arange(len(self.ecg_dict["Squared"]))[i]/256 + self.start_stamp for i in self.ecg_dict["Peaks"]],
                 [self.ecg_dict["Squared"][i] for i in self.ecg_dict["Peaks"]],
                 linestyle="", marker="v", color='black')

        ax4.set_ylabel("Voltage")
        ax4.set_xlabel("Seconds into collection")
        ax4.legend(loc='center right')

    def detect_ecg_peaks(self, start=None, stop=None):
        """Pan-Tompkins peak detection algorithm (1985).

        Code taken from ecgdetectors package (Luis Howell & Bernd Porr): https://github.com/luishowell/ecg-detectors

        :argument
        -start/stop: data cropping, seconds into collection
        """

        if start is not None and stop is not None:
            df = self.df.loc[(self.df["Timestamp"] >= start) & (self.df["Timestamp"] <= stop)]
        if start is None and stop is None:
            df = self.df

        self.start_stamp = start
        self.stop_stamp = stop

        # ECG PROCESSING ------------
        print("\nDetecting ECG peaks using Pan-Tompkins algorithm...")
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

        average_hr = (len(filt_peaks) - 1) / (df.shape[0] / 256) * 60
        print("-Found {} peaks in {} seconds.".format(len(filt_peaks), round(df.shape[0]/256), 1))
        print("    -Average HR is {} bpm.".format(round(average_hr, 1)))

        # Calcalates RR sd in ms
        rr_ints = [1000*(r2 - r1)/256 for r1, r2 in zip(filt_peaks[:], filt_peaks[1:])]
        rr_sd = np.std(rr_ints)

        print("    -SD of RR intervals is {} ms.".format(round(rr_sd, 1)))

        self.ecg_dict["Data"] = [i for i in df["ECG"]]
        self.ecg_dict["Data_Filt"] = [i for i in df["ECG_Filt"]]
        self.ecg_dict["Squared"] = filtered_squared
        self.ecg_dict["Wavelet"] = swt_ecg
        self.ecg_dict["Proc_Data"] = filtered_squared
        self.ecg_dict["Peaks"] = filt_peaks
        self.ecg_dict["RR ints"] = rr_ints
        self.ecg_dict["Beat HR"] = [round(1000 / r * 60, 1) for r in rr_ints]

    def plot_ppg_filter(self, start=None, stop=None):
        """Plots raw and filtered PPG data.

            :argument
            -start/stop: crop data by number of seconds.
        """

        print("\nPlotting raw and filtered PPG data...")

        if "PPG_Filt" not in self.df.columns:
            print("-Function requires filtered PPG data. Please run filtering function and try again.")
            return None

        if start is not None and stop is not None:
            df = self.df.loc[(self.df["Timestamp"] >= start) & (self.df["Timestamp"] <= stop)]
        if start is None and stop is None:
            df = self.df

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(self.fig_width, self.fig_height))

        if self.filter_dict["PPG_type"] == "bandpass":
            ax1.set_title("PPG Data: Raw vs. {}-{} Hz bandpass filtered".format(float(self.filter_dict["PPG_low"]),
                                                                          float(self.filter_dict["PPG_high"])))
        if self.filter_dict["PPG_type"] == "highpass":
            ax1.set_title("PPG Data: Raw vs. {} Hz highpass filtered".format(float(self.filter_dict["PPG_high"])))

        ax1.plot(np.arange(0, df.shape[0], 1) / 256 + start, df["PPG"], label="PPG_raw", color='black')
        ax1.legend()

        ax2.plot(np.arange(0, df.shape[0], 1) / 256 + start, df["PPG_Filt"], label="PPG_filt", color='red')
        ax2.plot([i / 256 for i in self.ppg_peaks], [df.iloc[i]["PPG_Filt"] for i in self.ppg_peaks],
                 linestyle="", marker="o", color='black', markersize=4)
        ax2.legend()

        ax2.set_xlabel("Seconds into collection")

    def plot_ppg_steps(self, start=None, stop=None):
        """Generates plots that shows steps involved in PPG peak detection algorithm.

               :argument
               -start/stop: crop data, number of seconds
        """

        print("\nPlotting PPG data processing steps...")

        if "PPG_Filt" not in self.df.columns:
            print("-Filtered PPG data is required. Please run filtering function and try again.")
            return None

        if start is None and stop is None:
            start = 0
            stop = int(self.df.shape[0]-5)

        df = self.df["PPG_Filt"].loc[(self.df["Timestamp"] >= start) &
                                     (self.df["Timestamp"] <= stop)]

        d = [p2 - p1 for p1, p2 in zip(df.iloc[:], df.iloc[1:])]
        d2 = [i * i for i in d]

        d2_filt = self.filter_signal(data=d2, filter_type="lowpass", low_f=3, sample_f=256)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(10, 7))
        plt.suptitle("PPG Data Processing Steps")

        ax1.plot(np.arange(0, df.shape[0]) / 256 + start, df, color='red', label="Filt")
        ax1.legend()
        ax2.plot(np.arange(0, df.shape[0] - 1) / 256 + start, d, color='blue', label="Diff")
        ax2.legend()
        ax3.plot(np.arange(0, df.shape[0] - 1) / 256 + start, d2, color='green', label="Diff sq.")
        ax3.legend()
        ax4.plot(np.arange(0, df.shape[0] - 1) / 256 + start, d2_filt, color='black', label="Diff sq. lowpass")

        peaks = peakutils.indexes(y=np.array(d2_filt), thres_abs=True, thres=350, min_dist=int(256 / 3))
        ax4.plot([peak / 256 + start for peak in peaks], [d2_filt[p] for p in peaks], linestyle="", marker="v",
                 color='dodgerblue')

        ax4.legend()
        ax4.set_xlabel("Seconds into collection")

    def process_ppg(self, show_plot=True, start=None, stop=None, remove_close_peaks=True,
                    use_heartpy=False, threshold=350):
        """Two peak detection methods for PPG data. Stores data in self.ppg_dict. Able to plot results.

            :argument
            -show_plot: boolean whether to show plot
            -start/stop: crop data by seconds
            -use_heartpy: boolean whether to use HeartPy package for peak detection. Will use more generic
                          peakutils method if False
                          -Alternative method involves filtering, differentiating, squaring, filtering, and running
                          peakutils on data.
            -remove_close_peaks: removes peaks that are too close (HR > 180bpm)
            -threshold: absolute threshold for peak detection used if use_heartpy = False
        """

        wd, m = None, None

        if start is not None and stop is not None:
            raw_data = np.asarray(self.df["PPG"].iloc[start*256:stop*256 - 2 * 256])
            filt_data = np.asarray(self.df["PPG_Filt"].iloc[start*256:stop*256 - 2 * 256])

        if start is None and stop is None:
            raw_data = np.asarray(self.df["PPG"].iloc[:-2 * 256])
            filt_data = np.asarray(self.df["PPG_Filt"].iloc[:-2*256])

        if use_heartpy:
            wd, m = heartpy.process(heartpy.scale_data(filt_data), 256)

            # Removes false peaks from wd['peakslist']
            peaks = [i for i in wd['peaklist']]
            rem_peaks = [i for i in wd["removed_beats"]]

            final_peaks = []
            for p in peaks:
                if p not in rem_peaks:
                    final_peaks.append(p)

            if remove_close_peaks:
                final_peaks2 = []
                for i in range(len(final_peaks) - 1):
                    diff = final_peaks[i + 1] - final_peaks[i]
                    if 256 * (60 / 40) >= diff >= 256 * (60 / 180):
                        final_peaks2.append(final_peaks[i])

                final_peaks = final_peaks2
                del final_peaks2

        if not use_heartpy:
            d = [p2 - p1 for p1, p2 in zip(filt_data[:], filt_data[1:])]
            d2 = [i * i for i in d]

            d2_filt = self.filter_signal(data=d2, filter_type="lowpass", low_f=3, sample_f=256)

            self.ppg_dict["Proc_Data"] = d2_filt

            final_peaks = peakutils.indexes(y=np.array(d2_filt), thres_abs=True, thres=threshold, min_dist=int(256/3))

            if remove_close_peaks:
                final_peaks2 = []
                for i in range(len(final_peaks) - 1):
                    diff = final_peaks[i + 1] - final_peaks[i]
                    if 256 * (60 / 40) >= diff >= 256 * (60 / 180):
                        final_peaks2.append(final_peaks[i])

                final_peaks = final_peaks2
                del final_peaks2

        self.ppg_dict["Data"] = [i for i in raw_data]
        self.ppg_dict["Data_Filt"] = [i for i in filt_data]
        self.ppg_dict["RR ints"] = [1000 * (r2 - r1) / 256 for r1, r2 in zip(final_peaks[:], final_peaks[1:])]
        self.ppg_dict["Beat HR"] = [round(1000 / r * 60, 1) for r in self.ppg_dict["RR ints"]]
        self.ppg_dict["Peaks"] = final_peaks

        print("\nRunning PPG peak detection...")
        print("-Found {} peaks in {} seconds.".format(len(final_peaks), round(len(filt_data)/256), 1))
        print("    -Average HR is {} bpm.".format(round(np.mean(self.ppg_dict["Beat HR"]), 1)))

        # Calcalates RR sd in ms
        rr_ints = [1000*(r2 - r1)/256 for r1, r2 in zip(final_peaks[:], final_peaks[1:])]
        rr_sd = np.std(rr_ints)

        print("    -SD of RR intervals is {} ms.".format(round(rr_sd, 1)))

        # PLOTTING ----------------------------------------------------------------------------------------------------
        if show_plot:
            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(self.fig_width, self.fig_height))

            ax1.plot([i / 256 for i in np.arange(len(raw_data))], raw_data, color='black', label="PPG_Raw")
            ax1.legend()

            ax2.plot([i / 256 for i in np.arange(len(filt_data))], filt_data, color='black', label="PPG_Filt")
            ax2.plot([np.arange(len(filt_data))[i] / 256 for i in final_peaks],
                     [filt_data[i] for i in final_peaks], linestyle="", color='dodgerblue', marker='v')
            ax2.legend()

            # beat to beat HR
            ax3.plot([i / 256 for i in final_peaks[:-1]], [1000 / i * 60 for i in self.ppg_dict["RR ints"]],
                     color='red', label="HR")
            ax3.legend()
            ax3.set_ylabel("HR (bpm)")
            ax3.set_xlabel("Seconds")
            plt.ylim(40, 180)

        return wd, m, final_peaks

    def compare_ecg_ppg_hr(self, start=None, stop=None, remove_close_ppg_peaks=True,
                           use_heartpy=False, threshold=350, plot_processed=False, show_plot=True):
        """Plots ECG and PPG data with detected peaks. Bottom plot overlays beat-to-beat HR calculated by each dataset.

            :argument
            -start/stop: crop data by seconds
            -remove_close_ppg_peaks: boolean whether to remove peaks that correspond to HR > 180bpm
            -use_heartpy: boolean whether to use HeartPy processing. Alternative peakutils method run if False
            -threshold: absolute threshold used in peakutils if use_heartpy = False
            -plot_processed: boolean whether to plot filtered or data that were processed for peak detection
        """

        self.detect_ecg_peaks(start=start, stop=stop)
        self.process_ppg(start=start, stop=stop, show_plot=False, threshold=threshold,
                         remove_close_peaks=remove_close_ppg_peaks, use_heartpy=use_heartpy)

        if show_plot:
            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(self.fig_width, self.fig_height))
            plt.subplots_adjust(hspace=.25)
            plt.suptitle("HR Comparison derived from ECG and PPG data")

            ax1.set_title("Filtered ECG")

            if plot_processed:
                ax1.plot(np.arange(0, len(self.ecg_dict["Proc_Data"])) / 256, self.ecg_dict["Proc_Data"],
                         color='black')
                ax1.plot([np.arange(len(self.ecg_dict["Proc_Data"]))[i] / 256 for i in self.ecg_dict["Peaks"]],
                         [self.ecg_dict["Proc_Data"][i] for i in self.ecg_dict["Peaks"]],
                         linestyle="", color='red', marker='v')

                ax2.plot(np.arange(0, len(self.ppg_dict["Proc_Data"])) / 256, self.ppg_dict["Proc_Data"],
                         color='dodgerblue')
                ax2.plot([np.arange(len(self.ppg_dict["Proc_Data"]))[i] / 256 for i in self.ppg_dict["Peaks"]],
                         [self.ppg_dict["Proc_Data"][i] for i in self.ppg_dict["Peaks"]],
                         linestyle="", color='black', marker='v')

            if not plot_processed:
                ax1.plot(np.arange(0, len(self.ecg_dict["Data_Filt"])) / 256, self.ecg_dict["Data_Filt"],
                         color='black')
                ax1.plot([np.arange(len(self.ecg_dict["Data_Filt"]))[i] / 256 for i in self.ecg_dict["Peaks"]],
                         [self.ecg_dict["Data_Filt"][i] for i in self.ecg_dict["Peaks"]],
                         linestyle="", color='red', marker='v')

                ax2.plot(np.arange(0, len(self.ppg_dict["Data_Filt"])) / 256, self.ppg_dict["Data_Filt"],
                         color='dodgerblue')
                ax2.plot([np.arange(len(self.ppg_dict["Data_Filt"]))[i] / 256 for i in self.ppg_dict["Peaks"]],
                         [self.ppg_dict["Data_Filt"][i] for i in self.ppg_dict["Peaks"]],
                         linestyle="", color='black', marker='v')

            ax1.set_ylabel("Voltage")
            ax2.set_title("Filtered PPG")

            ax3.set_title("Heart Rate")
            ax3.plot([np.arange(0, len(self.ppg_dict["Data_Filt"]))[i] / 256 for i in self.ppg_dict["Peaks"][:-1]],
                     self.ppg_dict["Beat HR"],
                     marker="o", color='dodgerblue',
                     label="PPG ({}bpm)".format(round(np.mean(self.ppg_dict["Beat HR"]), 1)), markersize=2)

            ax3.plot([np.arange(0, len(self.ecg_dict["Data_Filt"]))[i] / 256 for i in self.ecg_dict["Peaks"][:-1]],
                     self.ecg_dict["Beat HR"],
                     marker="s", color='black', label='ECG ({}bpm)'.format(round(np.mean(self.ecg_dict["Beat HR"]), 1)),
                     markersize=2)

            ax3.set_ylabel("HR (bpm)")
            ax3.legend()

    def plot_ecg_ppg_fft(self):
        """Plots FFT of entire file for ECG and PPG data. Runs .15Hz highpass filter to remove DC"""

        ppg = self.filter_signal(data=x.df["PPG"].dropna(), filter_type="highpass", high_f=.15, sample_f=256)
        ecg = self.filter_signal(data=x.df["ECG"].dropna(), filter_type="highpass", high_f=.15, sample_f=256)

        fft_ppg = scipy.fft.fft(ppg)
        fft_ecg = scipy.fft.fft(ecg)

        fs = 256
        l = x.df.shape[0]

        xf = np.linspace(0.0, 1.0 / (2.0 * (1 / fs)), l // 2)

        ppg_power = 2.0 / l / 2 * np.abs(fft_ppg[0:l // 2])
        ecg_power = 2.0 / l / 2 * np.abs(fft_ecg[0:l // 2])

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(self.fig_width, self.fig_height))
        plt.suptitle("Fast Fourier Transform Comparison "
                     "({}Hz highpass filter to remove DC)".format(self.filter_dict["PPG_high"]))

        ax1.plot(xf, ppg_power, color='dodgerblue', label="PPG")
        ax1.set_ylabel("Power")
        ax1.legend()

        ax2.plot(xf, ecg_power, color='red', label="ECG")
        ax2.set_ylabel("Power")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.legend()

        ax2.set_xlim(-1, 30)

    def plot_ecg_ppg_stft(self, multiplier=2, start=None, stop=None):
        """Calculates and plots Short-Time Fourier Transform for PPG and ECG data.

            :argument
            -start/stop: crop by seconds into collection
            -multiplier: I don't really get how STFTs work but a larger multiplier increase temporal accuracy while
                         reducing frequency resolution. Reverse is true for small multipliers
        """

        if start is not None and stop is not None:
            ecg = self.df.loc[(self.df["Timestamp"] >= start) & (self.df["Timestamp"] <= stop)]["ECG"]
            ppg = self.df.loc[(self.df["Timestamp"] >= start) & (self.df["Timestamp"] <= stop)]["PPG"]
        if start is None and stop is None:
            ecg = self.df["ECG"]
            ppg = self.df["PPG"]

        f, t, Zxx = scipy.signal.stft(x=ppg, fs=256, nperseg=int(256 / multiplier), window='hamming')

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(self.fig_width, self.fig_height))
        plt.subplots_adjust(hspace=.4)

        ax1.set_title("PPG Data")
        ax1.plot(np.arange(0, len(ppg)) / 256, ppg, color='black')

        ax2.pcolormesh(t, f, np.abs(Zxx))
        ax2.set_ylabel('Frequency [Hz]')
        ax2.set_ylim(-.1, 20)

        f, t, Zxx = scipy.signal.stft(x=ecg, fs=256, nperseg=int(256 / multiplier), window='hamming')

        ax3.plot(np.arange(0, len(ecg)) / 256, ecg, color='black')
        ax3.set_title("ECG Data")

        ax4.pcolormesh(t, f, np.abs(Zxx))
        ax4.set_ylabel('Frequency [Hz]')
        ax4.set_ylim(-.1, 20)
        ax4.set_xlabel('Seconds')

    def window_heart_rates(self, window_len=15):
        """Calculates and plots HR averaged over specified window length. Requires already-processed data."""

        if "Peaks" not in self.ppg_dict.keys() or "Peaks" not in self.ecg_dict.keys():
            print("-HR data is missing. Please process data and try again. ")
            return None

        ecg = self.ecg_dict["Peaks"]
        ppg = self.ppg_dict["Peaks"]

        window_indexes = np.arange(0, max(max(ecg), max(ppg)), window_len * 256)

        ecg_hr = []
        ppg_hr = []

        for window_num in range(len(window_indexes) - 1):
            e = [i for i in ecg if window_indexes[window_num] <= i <= window_indexes[window_num + 1]]
            p = [i for i in ppg if window_indexes[window_num] <= i <= window_indexes[window_num + 1]]

            ecg_beats = len(e)
            ppg_beats = len(p)
            ecg_time = (e[-1] - e[0]) / 256
            ppg_time = (p[-1] - p[0]) / 256

            ecg_hr.append(round((ecg_beats - 1) / ecg_time * 60, 1))
            ppg_hr.append(round((ppg_beats - 1) / ppg_time * 60, 1))

        self.ecg_dict["Windowed HR"] = ecg_hr
        self.ecg_dict["Window Length"] = window_len

        self.ppg_dict["Windowed HR"] = ppg_hr
        self.ppg_dict["Window Length"] = window_len

        # PLOTTING ---------------------------------------------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(2, figsize=(self.fig_width, self.fig_height))
        plt.subplots_adjust(hspace=.4)
        plt.suptitle("Comparison of ECG- and PPG-derived HR averaged over {}-second windows".format(window_len))

        # Time series HR
        ax1.set_title("Time Series Data")
        ax1.plot([i * window_len for i in range(len(ecg_hr))], ecg_hr, label='ECG',
                 color='black', marker="o", markersize=4)
        ax1.plot([i * window_len for i in range(len(ppg_hr))], ppg_hr, label='PPG',
                 color='dodgerblue', marker="o", markersize=4)
        ax1.legend()
        ax1.set_ylabel("HR (bpm)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylim(40, 180)

        # Bland-Altman comparison
        means = [(e + p) / 2 for e, p in zip(ecg_hr, ppg_hr)]
        diffs = [e - p for e, p in zip(ecg_hr, ppg_hr)]
        bias = round(np.mean(diffs), 1)
        sd = round(np.std(diffs), 1)
        ul = round(bias + 1.96*sd, 1)
        ll = round(bias - 1.96*sd, 1)

        ax2.set_title("Bland-Altman HR Comparison "
                      "({}/{} data points within limits)".format(len([d for d in diffs if ll <= d <= ul]), len(diffs)))

        ax2.scatter([means[i] for i in range(len(diffs)) if ll <= diffs[i] <= ul],
                    [diffs[i] for i in range(len(diffs)) if ll <= diffs[i] <= ul],
                    color='green', s=15)
        ax2.scatter([means[i] for i in range(len(diffs)) if ll > diffs[i] or diffs[i] > ul],
                    [diffs[i] for i in range(len(diffs)) if ll > diffs[i] or diffs[i] > ul],
                    color='red', marker="x", s=15)

        ax2.axhline(y=ul, linestyle='dashed', color='red',
                    label="Upper limit ({} bpm)".format(round(ul, 1)))
        ax2.axhline(y=bias,linestyle='dashed', color='black',
                    label='Bias ({} bpm)'.format(bias))
        ax2.axhline(y=ll, linestyle='dashed', color='red',
                    label="Lower limit ({} bpm)".format(round(ll, 1)))

        xlims = ax2.get_xlim()
        ax2.set_xlim(xlims[0], xlims[1]+5)
        ylims = ax2.get_ylim()
        ax2.set_ylim(ylims[0]-5, ylims[1]+3)

        ax2.fill_between(x=[ax2.get_xlim()[0], ax2.get_xlim()[1]], y1=ul, y2=ul+20, color='red', alpha=.10)
        ax2.fill_between(x=[ax2.get_xlim()[0], ax2.get_xlim()[1]], y1=ll, y2=ll-20, color='red', alpha=.10)
        ax2.fill_between(x=[ax2.get_xlim()[0], ax2.get_xlim()[1]], y1=ll, y2=ul, color='green', alpha=.10)

        ax2.text(x=max(means)+1, y=bias+1, s="Bias = {} bpm".format(bias), fontsize=12)
        ax2.text(x=max(means)+1, y=ul+1, s="Up Lim = {} bpm".format(ul), fontsize=12, color='red')
        ax2.text(x=max(means)+1, y=ll-4, s="Low Lim = {} bpm".format(ll), fontsize=12, color='red')

        ax2.set_ylabel("Δ bpm (ECG - PPG)")
        ax2.set_xlabel("Mean HR (bpm)")

        # ax2.legend(loc='best')

    def compare_accel_gyro(self, start=None, stop=None, use_filtered=False):

        if start is not None and stop is not None:
            df = self.df.loc[(self.df["Timestamp"] >= start) & (self.df["Timestamp"] <= stop)]
        if start is None and stop is None:
            df = self.df

        if use_filtered:
            col_suffix = "_Filt"
        if not use_filtered:
            col_suffix = ""

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(self.fig_width, self.fig_height))

        ax1.set_title("Accelerometer" + col_suffix)
        ax1.plot(np.arange(0, df.shape[0]) / 256, df["Accel_X" + col_suffix], color='black', label="X")
        ax1.plot(np.arange(0, df.shape[0]) / 256, df["Accel_Y" + col_suffix], color='red', label="Y")
        ax1.plot(np.arange(0, df.shape[0]) / 256, df["Accel_Z" + col_suffix], color='dodgerblue', label="Z")
        ax1.legend()
        ax1.set_ylabel("G")

        ax2.set_title("Gyroscope" + col_suffix)
        ax2.plot(np.arange(0, df.shape[0]) / 256, df["Gyro_X" + col_suffix], color='black', label="X")
        ax2.plot(np.arange(0, df.shape[0]) / 256, df["Gyro_Y" + col_suffix], color='red', label="Y")
        ax2.plot(np.arange(0, df.shape[0]) / 256, df["Gyro_Z" + col_suffix], color='dodgerblue', label="Z")
        ax2.legend()
        ax2.set_ylabel("Degrees per second")

        ax2.set_xlabel("Seconds")


x = Data()

# ================================================= PPG-only data =====================================================

# Filters data
x.df["PPG_Filt"] = x.filter_signal(data=x.df["PPG"], filter_type="bandpass", low_f=0.67, high_f=12, sample_f=256)

# Plots raw and filtered PPG data
# x.plot_ppg_filter(start=15, stop=45)

# Plots peak detection processing steps in specified region
# x.plot_ppg_steps(start=15, stop=60)

# x.process_ppg(start=5, stop=280, show_plot=True, use_heartpy=False, threshold=350, remove_close_peaks=True)

# ================================================= ECG-only data =====================================================

# Filters ECG data
x.df["ECG_Filt"] = x.filter_signal(data=x.df["ECG"], filter_type="bandpass", low_f=3, high_f=18, data_type="ECG")

# Finds peaks and plots raw + filtered data with peaks + beat-to-beat HR in specified region
# Prints number of peaks found, average HR and RR SD
# x.ecg_find_and_plot_peaks(start=5, stop=280, show_plot=True)

# Plots peak detection processing steps in specified region
# x.plot_ecg_steps(start=30, stop=60)

# ================================================ ECG and PPG data ===================================================

# Data, peak, and beat-by-beat HR comparison between PPG and ECG
# x.compare_ecg_ppg_hr(start=5, stop=280, remove_close_ppg_peaks=True, use_heartpy=False, threshold=350, plot_processed=False, show_plot=False)

# Plots FFT of entire file for raw PPG and ECG data
# x.plot_ecg_ppg_fft()

# Plots STFT for PPG and ECG data for specified region
# x.plot_ecg_ppg_stft(multiplier=10, start=5, stop=25)

# Plots HR averaged over specified window
# x.window_heart_rates(window_len=30)

# ============================================== Accel and Gyro data ==================================================

# Filters accelerometer and gyroscope data
x.filter_imu_signal(data_type="accel", filter_type="highpass", high_f=.1, sample_f=256)
x.filter_imu_signal(data_type="gyro", filter_type="bandpass", low_f=0.5, high_f=3, sample_f=256)

# Plots raw or filtered gyroscope and accelerometer data in specified region
x.compare_accel_gyro(start=5, stop=20, use_filtered=False)

