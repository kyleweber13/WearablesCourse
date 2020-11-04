from ecgdetectors import Detectors
# https://github.com/luishowell/ecg-detectors

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statistics
import scipy.stats as stats
from datetime import datetime
import matplotlib.dates as mdates
import pyedflib
from datetime import timedelta
import os
import math
import sys


class ECG:

    def __init__(self, filepath=None, downsample_ratio=2, age=33, rest_hr_window=60, n_epochs_rest=5, epoch_len=15,
                 fig_width=10, fig_height=7):

        print()
        print("============================================= ECG DATA ==============================================")

        self.filepath = filepath
        self.age = age
        self.epoch_len = epoch_len
        self.rest_hr_window = rest_hr_window
        self.n_epochs_rest = n_epochs_rest
        self.downsample_ratio = downsample_ratio

        self.fig_width = fig_width
        self.fig_height = fig_height

        self.df_raw = None
        self.df_accel = None
        self.df_epoch = None

        self.sample_rate = 250
        self.accel_sample_rate = 25
        self.rest_hr = 55

        # Raw data
        self.timestamps, self.raw, self.accel_vm = self.import_file()

        self.hr_max = 208 - .7 * self.age  # Tanaka et al., 2001 equation

        # Dictionary for HR that correspond to %HRR zones
        self.hr_zones = {"Sedentary": 0}

    def import_file(self):
        """Method that loads voltage channel, sample rate, starttime, and file duration.
        Creates timestamp for each data point."""

        t0 = datetime.now()

        print("\n" + "Importing {}...".format(self.filepath))

        file = pyedflib.EdfReader(self.filepath)

        self.sample_rate = file.getSampleFrequencies()[0]

        # READS IN ECG DATA ===========================================================================================
        print("Importing ECG file...")
        raw = file.readSignal(chn=0)

        print("ECG data import complete.")

        starttime = file.getStartdatetime()

        # DOWNSAMPLING
        if self.downsample_ratio != 1:
            raw = raw[::self.downsample_ratio]
            self.sample_rate = int(self.sample_rate / self.downsample_ratio)

        # TIMESTAMP GENERATION
        t0_stamp = datetime.now()

        print("\n" + "Creating timestamps...")

        # Timestamps
        end_time = starttime + timedelta(seconds=len(raw) / self.sample_rate)
        timestamps = np.asarray(pd.date_range(start=starttime, end=end_time, periods=len(raw)))

        t1_stamp = datetime.now()
        stamp_time = (t1_stamp - t0_stamp).seconds
        print("Complete ({} seconds).".format(round(stamp_time, 2)))

        t1 = datetime.now()
        proc_time = (t1 - t0).seconds
        print("\n" + "Import complete ({} seconds).".format(round(proc_time, 2)))

        # ACCELEROMETER DATA ==========================================================================================
        self.accel_sample_rate = file.getSampleFrequencies()[1]

        x = file.readSignal(chn=1)
        y = file.readSignal(chn=2)
        z = file.readSignal(chn=3)

        if self.accel_sample_rate == 100:
            x = x[::4]
            y = y[::4]
            z = z[::4]

            self.accel_sample_rate = 25

        vm = (np.sqrt(np.square(np.array([x, y, z])).sum(axis=0)) - 1000) / 1000
        vm[vm < 0] = 0

        # file.close()

        return timestamps, raw, vm

    def check_quality(self):
        """Performs quality check using Orphanidou et al. (2015) algorithm that has been tweaked to factor in voltage
           range as well.

           This function runs a loop that creates object from the class CheckQuality for each epoch in the raw data.
        """

        print("\n" + "Running ECG signal quality check with Orphanidou et al. (2015) algorithm...")

        t0 = datetime.now()

        validity_list = []  # window's validity (binary; 1 = invalid)
        epoch_hr = []  # window's HRs

        raw = [i for i in self.df_raw["Raw"]]

        for start_index in range(0, len(raw), self.epoch_len * self.sample_rate):

            qc = CheckQuality(ecg_object=self, start_index=start_index, epoch_len=self.epoch_len)

            if qc.valid_period:
                validity_list.append("Valid")
                epoch_hr.append(round(qc.hr, 2))

            if not qc.valid_period:
                validity_list.append("Invalid")
                epoch_hr.append(0)

        t1 = datetime.now()
        proc_time = (t1 - t0).seconds
        print("\n" + "Quality check complete ({} seconds).".format(round(proc_time, 2)))
        print("-Processing time of {} seconds per "
              "hour of data.".format(round(proc_time / (self.df_raw.shape[0]/self.sample_rate/3600)), 2))

        # List of epoched heart rates but any invalid epoch is marked as None instead of 0 (as is self.epoch_hr)
        valid_hr = [epoch_hr[i] if validity_list[i] == "Valid"
                    else None for i in range(len(epoch_hr))]

        vm = [i for i in self.df_accel["VM"]]
        svm = [sum(vm[i:i+self.epoch_len*self.accel_sample_rate]) for
               i in np.arange(0, len(vm), self.epoch_len * self.accel_sample_rate)]

        self.df_epoch = pd.DataFrame(list(zip(self.df_raw["Timestamp"].iloc[::self.sample_rate * self.epoch_len],
                                              validity_list, epoch_hr, valid_hr, svm)),
                                     columns=["Timestamp", "Validity", "HR", "Valid HR", "SVM"])

    def find_resting_hr(self, window_size, n_windows):
        """Function that calculates resting HR based on inputs.

        :argument
        -window_size: size of window over which rolling average is calculated, seconds
        -n_windows: number of epochs over which resting HR is averaged (lowest n_windows number of epochs)
        -sleep_status: data from class Sleep that corresponds to asleep/awake epochs
        """

        epoch_hr = np.array(self.df_epoch["HR"])

        # Sets integer for window length based on window_size and epoch_len
        window_len = int(window_size / self.epoch_len)

        # Calculates rolling average HR; ignores 0s which are invalid period
        try:
            rolling_avg = [statistics.mean(epoch_hr[i:i + window_len]) if 0 not in epoch_hr[i:i + window_len]
                           else None for i in range(len(epoch_hr))]
        except statistics.StatisticsError:
            print("No data points found.")
            rolling_avg = []

        # Calculates resting HR during all hours if sleep_log not available -------------------------------------------
        valid_hr = [i for i in rolling_avg if i is not None]
        sorted_hr = sorted(valid_hr)

        self.rest_hr = round(sum(sorted_hr[:n_windows]) / n_windows, 1)

        print("Resting HR (average of {} lowest {}-second periods) is {} bpm.".format(n_windows,
                                                                                      window_size,
                                                                                      self.rest_hr))

    def calculate_percent_hrr(self):

        hrr = (208 - 0.7 * self.age) - self.rest_hr

        self.df_epoch["HRR"] = [(i - self.rest_hr) / hrr * 100 if i is not None else None
                                for i in self.df_epoch["Valid HR"]]

        self.hr_zones["Light"] = round(30 * hrr / 100 + self.rest_hr, 1)
        self.hr_zones["Moderate"] = round(40 * hrr / 100 + self.rest_hr, 1)
        self.hr_zones["Vigorous"] = round(60 * hrr / 100 + self.rest_hr, 1)

    def plot_hr_zones(self):

        fig, ax1 = plt.subplots(1, figsize=(self.fig_width, self.fig_height))
        plt.title("Time series HR ({}-second average)".format(self.epoch_len))
        plt.plot(self.df_epoch["Timestamp"], self.df_epoch["Valid HR"], color='black')

        plt.fill_between(x=self.df_epoch["Timestamp"], y1=0, y2=self.hr_zones["Light"],
                         color='grey', alpha=.3, label="Sedentary")

        plt.fill_between(x=self.df_epoch["Timestamp"], y1=self.hr_zones["Light"], y2=self.hr_zones["Moderate"],
                         color='green', alpha=.3, label="Light")

        plt.fill_between(x=self.df_epoch["Timestamp"], y1=self.hr_zones["Moderate"], y2=self.hr_zones["Vigorous"],
                         color='orange', alpha=.3, label="Moderate")

        plt.fill_between(x=self.df_epoch["Timestamp"], y1=self.hr_zones["Vigorous"], y2=self.hr_max,
                         color='red', alpha=.3, label="Vigorous")

        plt.ylabel("HR (bpm)")
        plt.legend()
        plt.ylim(40, self.hr_max)
        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
        ax1.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)


class CheckQuality:
    """Class method that implements the Orphanidou ECG signal quality assessment algorithm on raw ECG data.

       Orphanidou, C. et al. (2015). Signal-Quality Indices for the Electrocardiogram and Photoplethysmogram:
       Derivation and Applications to Wireless Monitoring. IEEE Journal of Biomedical and Health Informatics.
       19(3). 832-838.
    """

    def __init__(self, ecg_object, start_index, voltage_thresh=250, epoch_len=15):
        """Initialization method.

        :param
        -ecg_object: EcgData class instance created by ImportEDF script
        -random_data: runs algorithm on randomly-generated section of data; False by default.
                      Takes priority over start_index.
        -start_index: index for windowing data; 0 by default
        -epoch_len: window length in seconds over which algorithm is run; 15 seconds by default
        """

        self.voltage_thresh = voltage_thresh
        self.epoch_len = epoch_len
        self.fs = ecg_object.sample_rate
        self.start_index = start_index

        self.ecg_object = ecg_object

        self.raw_data = [i for i in ecg_object.df_raw["Raw"].iloc[self.start_index:
                                                                  self.start_index + self.epoch_len * self.fs]]

        self.index_list = np.arange(0, len(self.raw_data), self.epoch_len * self.fs)

        self.rule_check_dict = {"Valid Period": False,
                                "HR Valid": False, "HR": None,
                                "Max RR Interval Valid": False, "Max RR Interval": None,
                                "RR Ratio Valid": False, "RR Ratio": None,
                                "Voltage Range Valid": False, "Voltage Range": None,
                                "Correlation Valid": False, "Correlation": None,
                                "Accel Counts": None}

        # prep_data parameters
        self.r_peaks = None
        self.r_peaks_index_all = None
        self.rr_sd = None
        self.removed_peak = []
        self.enough_beats = True
        self.hr = 0
        self.delta_rr = []
        self.removal_indexes = []
        self.rr_ratio = None
        self.volt_range = 0

        # apply_rules parameters
        self.valid_hr = None
        self.valid_rr = None
        self.valid_ratio = None
        self.valid_range = None
        self.valid_corr = None
        self.rules_passed = None

        # adaptive_filter parameters
        self.median_rr = None
        self.ecg_windowed = []
        self.average_qrs = None
        self.average_r = 0

        # calculate_correlation parameters
        self.beat_ppmc = []
        self.valid_period = None

        """RUNS METHODS"""
        # Peak detection and basic outcome measures
        self.prep_data()

        # Runs rules check if enough peaks found
        if self.enough_beats:
            self.adaptive_filter()
            self.calculate_correlation()
            self.apply_rules()

    def prep_data(self):
        """Function that:
        -Initializes ecgdetector class instance
        -Runs stationary wavelet transform peak detection
            -Implements 0.1-10Hz bandpass filter
            -DB3 wavelet transformation
            -Pan-Tompkins peak detection thresholding
        -Calculates RR intervals
        -Removes first peak if it is within median RR interval / 2 from start of window
        -Calculates average HR in the window
        -Determines if there are enough beats in the window to indicate a possible valid period
        """

        # Initializes Detectors class instance with sample rate
        detectors = Detectors(self.fs)

        # Runs peak detection on raw data ----------------------------------------------------------------------------
        # Uses ecgdetectors package -> stationary wavelet transformation + Pan-Tompkins peak detection algorithm
        self.r_peaks = detectors.swt_detector(unfiltered_ecg=self.raw_data)

        if len(self.r_peaks) == 3:
            print("\nYou need to unedit ecgdetectors package.")
            sys.exit()

        # Checks to see if there are enough potential peaks to correspond to correct HR range ------------------------
        # Requires number of beats in window that corresponds to ~40 bpm to continue
        # Prevents the math in the self.hr calculation from returning "valid" numbers with too few beats
        # i.e. 3 beats in 3 seconds (HR = 60bpm) but nothing detected for rest of epoch
        if len(self.r_peaks) >= np.floor(40/60*self.epoch_len):
            self.enough_beats = True

            n_beats = len(self.r_peaks)  # number of beats in window
            delta_t = (self.r_peaks[-1] - self.r_peaks[0]) / self.fs  # time between first and last beat, seconds
            self.hr = 60 * (n_beats-1) / delta_t  # average HR, bpm

        # Stops function if not enough peaks found to be a potential valid period
        # Threshold corresponds to number of beats in the window for a HR of 40 bpm
        if len(self.r_peaks) < np.floor(40/60*self.epoch_len):
            self.enough_beats = False
            self.valid_period = False
            return

        # Calculates RR intervals in seconds -------------------------------------------------------------------------
        for peak1, peak2 in zip(self.r_peaks[:], self.r_peaks[1:]):
            rr_interval = (peak2 - peak1) / self.fs
            self.delta_rr.append(rr_interval)

        # Approach 1: median RR characteristics ----------------------------------------------------------------------
        # Calculates median RR-interval in seconds
        median_rr = np.median(self.delta_rr)

        # SD of RR intervals in ms
        self.rr_sd = np.std(self.delta_rr) * 1000

        # Converts median_rr to samples
        self.median_rr = int(median_rr * self.fs)

        # Removes any peak too close to start/end of data section: affects windowing later on ------------------------
        # Peak removed if within median_rr/2 samples of start of window
        # Peak removed if within median_rr/2 samples of end of window
        for i, peak in enumerate(self.r_peaks):
            # if peak < (self.median_rr/2 + 1) or (self.epoch_len*self.fs - peak) < (self.median_rr/2 + 1):
            if peak < (self.median_rr / 2 + 1) or (self.epoch_len * self.fs - peak) < (self.median_rr / 2 + 1):
                self.removed_peak.append(self.r_peaks.pop(i))
                self.removal_indexes.append(i)

        # Removes RR intervals corresponding to
        if len(self.removal_indexes) != 0:
            self.delta_rr = [self.delta_rr[i] for i in range(len(self.r_peaks)) if i not in self.removal_indexes]

        # Calculates range of ECG voltage ----------------------------------------------------------------------------
        self.volt_range = max(self.raw_data) - min(self.raw_data)

    def adaptive_filter(self):
        """Method that runs an adaptive filter that generates the "average" QRS template for the window of data.

        - Calculates the median RR interval
        - Generates a sub-window around each peak, +/- RR interval/2 in width
        - Deletes the final beat sub-window if it is too close to end of data window
        - Calculates the "average" QRS template for the window
        """

        # Approach 1: calculates median RR-interval in seconds  -------------------------------------------------------
        # See previous method

        # Approach 2: takes a window around each detected R-peak of width peak +/- median_rr/2 ------------------------
        for peak in self.r_peaks:
            window = self.raw_data[peak - int(self.median_rr / 2):peak + int(self.median_rr / 2)]

            self.ecg_windowed.append(window)  # Adds window to list of windows

        # Approach 3: determine average QRS template ------------------------------------------------------------------
        self.ecg_windowed = np.asarray(self.ecg_windowed)[1:]  # Converts list to np.array; omits first empty array

        # Calculates "correct" length (samples) for each window (median_rr number of datapoints)
        correct_window_len = 2*int(self.median_rr/2)

        # Removes final beat's window if its peak is less than median_rr/2 samples from end of window
        # Fixes issues when calculating average_qrs waveform
        if len(self.ecg_windowed[-1]) != correct_window_len:
            self.removed_peak.append(self.r_peaks.pop(-1))
            self.ecg_windowed = self.ecg_windowed[:-2]

        # Calculates "average" heartbeat using windows around each peak
        try:
            self.average_qrs = np.mean(self.ecg_windowed, axis=0)
        except ValueError:
            print("Failed to calculate mean QRS template.")

    def calculate_correlation(self):
        """Method that runs a correlation analysis for each beat and the average QRS template.

        - Runs a Pearson correlation between each beat and the QRS template
        - Calculates the average individual beat Pearson correlation value
        - The period is deemed valid if the average correlation is >= 0.66, invalid is < 0.66
        """

        # Calculates correlation between each beat window and the average beat window --------------------------------
        for beat in self.ecg_windowed:
            r = stats.pearsonr(x=beat, y=self.average_qrs)
            self.beat_ppmc.append(abs(r[0]))

        self.average_r = float(np.mean(self.beat_ppmc))
        self.average_r = round(self.average_r, 3)

    def apply_rules(self):
        """First stage of algorithm. Checks data against three rules to determine if the window is potentially valid.
        -Rule 1: HR needs to be between 40 and 180bpm
        -Rule 2: no RR interval can be more than 3 seconds
        -Rule 3: the ratio of the longest to shortest RR interval is less than 2.2
        -Rule 4: the amplitude range of the raw ECG voltage must exceed n microV (approximate range for non-wear)
        -Rule 5: the average correlation coefficient between each beat and the "average" beat must exceed 0.66
        -Verdict: all rules need to be passed
        """

        # Rule 1: "The HR extrapolated from the sample must be between 40 and 180 bpm" -------------------------------
        if 40 <= self.hr <= 180:
            self.valid_hr = True
        else:
            self.valid_hr = False

        # Rule 2: "the maximum acceptable gap between successive R-peaks is 3s ---------------------------------------
        for rr_interval in self.delta_rr:
            if rr_interval < 3:
                self.valid_rr = True

            if rr_interval >= 3:
                self.valid_rr = False
                break

        # Rule 3: "the ratio of the maximum beat-to-beat interval to the minimum beat-to-beat interval... ------------
        # should be less than 2.5"
        self.rr_ratio = max(self.delta_rr) / min(self.delta_rr)

        if self.rr_ratio >= 2.5:
            self.valid_ratio = False

        if self.rr_ratio < 2.5:
            self.valid_ratio = True

        # Rule 4: the range of the raw ECG signal needs to be >= 250 microV ------------------------------------------
        if self.volt_range <= self.voltage_thresh:
            self.valid_range = False

        if self.volt_range > self.voltage_thresh:
            self.valid_range = True

        # Rule 5: Determines if average R value is above threshold of 0.66 -------------------------------------------
        if self.average_r >= 0.66:
            self.valid_corr = True

        if self.average_r < 0.66:
            self.valid_corr = False

        # FINAL VERDICT: valid period if all rules are passed --------------------------------------------------------
        if self.valid_hr and self.valid_rr and self.valid_ratio and self.valid_range and self.valid_corr:
            self.valid_period = True
        else:
            self.valid_period = False

        self.rule_check_dict = {"Valid Period": self.valid_period,
                                "HR Valid": self.valid_hr, "HR": round(self.hr, 1),
                                "Max RR Interval Valid": self.valid_rr, "Max RR Interval": round(max(self.delta_rr), 1),
                                "RR Ratio Valid": self.valid_ratio, "RR Ratio": round(self.rr_ratio, 1),
                                "Voltage Range Valid": self.valid_range, "Voltage Range": round(self.volt_range, 1),
                                "Correlation Valid": self.valid_corr, "Correlation": self.average_r}


class Accel:

    def __init__(self, leftwrist_filepath=None, leftankle_filepath=None, epoch_len=15, fig_height=7, fig_width=12):

        print("\n===================================== ACCELEROMETER DATA ==========================================")

        # Default values for objects ----------------------------------------------------------------------------------
        self.lankle_fname = leftankle_filepath
        self.lwrist_fname = leftwrist_filepath
        self.epoch_len = epoch_len
        self.fig_height = fig_height
        self.fig_width = fig_width

        self.df_epoch = None  # dataframe of all devices for one epoch length

        self.activity_volume = None  # activity volume for one epoch length
        self.activity_df = None

        # Methods and objects that are run automatically when class instance is created -------------------------------

        self.df_lankle, self.lankle_samplerate = self.load_correct_file(filepath=self.lankle_fname,
                                                                        f_type="Left Ankle")
        self.df_lwrist, self.lwrist_samplerate = self.load_correct_file(filepath=self.lwrist_fname,
                                                                        f_type="Left Wrist")

        # Scaled cutpoints for 1-second epoch
        self.cutpoint_dict = {"NonDomLight": round(47 * self.lwrist_samplerate / 30 / 15, 2),
                              "NonDomModerate": round(64 * self.lwrist_samplerate / 30 / 15, 2),
                              "NonDomVigorous": round(157 * self.lwrist_samplerate / 30 / 15, 2),
                              "Epoch length": 1}

        self.recalculate_epoch_len(epoch_len=self.epoch_len)

    def load_correct_file(self, filepath, f_type) -> object:
        """Method that specifies the correct file (.edf vs. .csv) to import for accelerometer files and
           retrieves sample rates.
        """

        if filepath is None:
            print("\nNo {} filepath given.".format(f_type))
            return None, None

        if ".csv" in filepath or ".CSV" in filepath:
            df, sample_rate = self.import_csv(filepath, f_type=f_type)

        return df, sample_rate

    @staticmethod
    def import_csv(filepath=None, f_type="Accelerometer"):
        """Method to import GENEActiv .csv files. Only works for .csv. files created using GENEAtiv software.
           Finds sampling rate from data file.
        """

        if filepath is None:
            print("No {} filepath given.".format(f_type))
            return None, None

        if filepath is not None and not os.path.exists(filepath):
            print("Invalid {} filepath given. Try again.".format(f_type))
            return None, None

        t0 = datetime.now()
        print("\nImporting {}...".format(filepath))

        sample_rate = pd.read_csv(filepath_or_buffer=filepath, nrows=10, sep=",")
        sample_rate.columns = ["Variable", "Value"]
        sample_rate = float(sample_rate.loc[sample_rate["Variable"] == "Measurement Frequency"]["Value"].
                            iloc[0].split(" ")[0])

        df = pd.read_csv(filepath_or_buffer=filepath, skiprows=99, sep=",")
        df.columns = ["Timestamp", "Mean_X", "Mean_Y", "Mean_Z", "Mean_Lux", "Sum_Button",
                      "Mean_Temp", "SVM", "SD_X", "SD_Y", "SD_Z", "Peak_Lux"]
        df = df[["Timestamp", "SVM"]]

        df["Timestamp"] = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S:%f") for i in df["Timestamp"]]

        t1 = datetime.now()
        proc_time = (t1 - t0).seconds
        print("Import complete ({} seconds).".format(round(proc_time, 2)))

        return df, sample_rate

    def recalculate_epoch_len(self, epoch_len=15):

        print("\nRecalculating epoch length to {} seconds...".format(epoch_len))

        # Whole data frames if no cropping
        df_lankle = self.df_lankle
        df_lwrist = self.df_lwrist

        # Empty lists as placeholders for missing data
        lankle_epoched = [None for i in range(df_lankle.shape[0])]
        lwrist_epoched = [None for i in range(df_lwrist.shape[0])]

        timestamps_found = False

        # Re-calculating SVMs ----------------------------------------------------------------------------------------
        if self.df_lankle is not None:
            timestamps = df_lankle["Timestamp"].iloc[::epoch_len]

            timestamps_found = True
            df_timestamps = timestamps

            svm = [i for i in df_lankle["SVM"]]

            lankle_epoched = [sum(svm[i:i + epoch_len]) for i in range(0, df_lankle.shape[0], epoch_len)]

        if self.df_lwrist is not None:
            timestamps = df_lwrist["Timestamp"].iloc[::epoch_len]

            if not timestamps_found:
                df_timestamps = timestamps

            svm = [i for i in df_lwrist["SVM"]]

            lwrist_epoched = [sum(svm[i:i + epoch_len]) for i in range(0, self.df_lwrist.shape[0], epoch_len)]

        # Combines all devices' counts into one dataframe
        self.df_epoch = pd.DataFrame(list(zip(df_timestamps, lankle_epoched, lwrist_epoched)),
                                       columns=["Timestamp", "LAnkle", "LWrist"])

        # Scales cutpoints
        self.cutpoint_dict = {"NonDomLight": self.cutpoint_dict["NonDomLight"] *
                                             (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "NonDomModerate": self.cutpoint_dict["NonDomModerate"] *
                                                (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "NonDomVigorous": self.cutpoint_dict["NonDomVigorous"] *
                                                (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "Epoch length": epoch_len}

        print("Done.")


class Subject:

    def __init__(self, ecg_filepath=None, leftwrist_filepath=None, leftankle_filepath=None, epoch_len=15,
                 ecg_downsample_ratio=2, age=33, rest_hr_window=60, n_epochs_rest=5, fig_height=7, fig_width=12):

        self.ecg_filepath = ecg_filepath
        self.lwrist_filepath = leftwrist_filepath
        self.lankle_filepath = leftankle_filepath

        self.epoch_len = epoch_len
        self.downsample_ratio = ecg_downsample_ratio
        self.age = age
        self.rest_hr_window = rest_hr_window
        self.n_epochs_rest = n_epochs_rest

        self.df_epoch = None

        self.fig_height = fig_height
        self.fig_width = fig_width

        # ECG data object
        self.ecg = ECG(filepath=self.ecg_filepath, downsample_ratio=self.downsample_ratio, epoch_len=self.epoch_len)

        # Accelerometer data object
        self.accel = Accel(leftwrist_filepath=self.lwrist_filepath,
                           leftankle_filepath=self.lankle_filepath)

        # Syncs data
        self.sync_epochs()

        # More ECG processing
        # Finds time periods of unusable data
        self.ecg.check_quality()

        # Calculates resting and max HR
        self.ecg.find_resting_hr(window_size=self.ecg.rest_hr_window, n_windows=self.ecg.n_epochs_rest)

        self.ecg.calculate_percent_hrr()

        # Combines cropped accel and ECG data into one df
        self.crop_df_epoch()

        self.activity_volume = None
        self.activity_df = None
        self.hr_epoch = None

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

    def sync_epochs(self):
        """Crops raw data from ECG and 1s epoched accelerometer data so files start at same time."""

        print("\nSyncing raw data so epochs align...")

        # Finds start/stop times
        start = max(self.accel.df_epoch["Timestamp"].iloc[0], self.ecg.timestamps[0])
        stop = min(self.accel.df_epoch["Timestamp"].iloc[-1], self.ecg.timestamps[-1])

        # ECG data
        hr = pd.DataFrame(list(zip(self.ecg.timestamps, self.ecg.raw)), columns=["Timestamp", "Raw"])
        hr = hr.loc[(hr["Timestamp"] >= start) & (hr["Timestamp"] <= stop)]

        # Bittium accel data
        bf_accel = pd.DataFrame(list(zip(self.ecg.timestamps[::int(self.ecg.sample_rate / self.ecg.accel_sample_rate)],
                                         self.ecg.accel_vm)), columns=["Timestamp", "VM"])
        bf_accel = bf_accel.loc[(bf_accel["Timestamp"] >= start) & (bf_accel["Timestamp"] <= stop)]

        self.ecg.df_raw = hr
        self.ecg.df_accel = bf_accel
        del self.ecg.timestamps, self.ecg.raw, self.ecg.accel_vm

        # GENEActiv data
        self.accel.df_epoch = self.accel.df_epoch.loc[(self.accel.df_epoch["Timestamp"] >= start) &
                                                      (self.accel.df_epoch["Timestamp"] <= stop)]

        print("Complete.")

    def crop_df_epoch(self):
        """Crops and combines epoched data together."""

        print("\nCropping dataframes so wrist and HR data is from same time period...")

        start = max(self.accel.df_epoch["Timestamp"].iloc[0], self.ecg.df_epoch["Timestamp"].iloc[0])
        stop = min(self.accel.df_epoch["Timestamp"].iloc[-1], self.ecg.df_epoch["Timestamp"].iloc[-1])

        df_accel = self.accel.df_epoch.loc[(self.accel.df_epoch["Timestamp"] >= start) &
                                           (self.accel.df_epoch["Timestamp"] <= stop)]

        df_ecg = self.ecg.df_epoch.loc[(self.ecg.df_epoch["Timestamp"] >= start) &
                                       (self.ecg.df_epoch["Timestamp"] <= stop)]

        df = pd.DataFrame(list(zip(df_accel["Timestamp"], df_accel["LAnkle"], df_accel["LWrist"],
                                   df_ecg["Validity"], df_ecg["HR"], df_ecg["Valid HR"], df_ecg["HRR"], df_ecg["SVM"])),
                          columns=["Timestamp", "LAnkle", "LWrist", "HR Validity", "HR", "Valid HR", "HRR", "Chest"])

        print("Complete.")

        self.df_epoch = df

    def plot_hr_wrist(self):
        """Plots LWrist and HR data. Marks intensity regions
           (HR = %HRR ranges; LWrist = Powell et al. 2017 cutpoints.
        """

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(self.fig_width, self.fig_height))

        plt.suptitle("HR and Wrist Activity Count Data")

        # HR DATA ====================================================================================================

        ax1.set_title("Heart Rate")
        ax1.plot(self.df_epoch["Timestamp"], self.df_epoch["Valid HR"], color='black')
        ax1.set_ylabel("HR")

        ax1.fill_between(x=self.df_epoch["Timestamp"], y1=0, y2=self.ecg.hr_zones["Light"],
                         color='grey', alpha=.3, label="Sedentary")

        ax1.fill_between(x=self.df_epoch["Timestamp"],
                         y1=self.ecg.hr_zones["Light"], y2=self.ecg.hr_zones["Moderate"],
                         color='green', alpha=.3, label="Light")

        ax1.fill_between(x=self.df_epoch["Timestamp"],
                         y1=self.ecg.hr_zones["Moderate"], y2=self.ecg.hr_zones["Vigorous"],
                         color='orange', alpha=.3, label="Moderate")

        ax1.fill_between(x=self.df_epoch["Timestamp"], y1=self.ecg.hr_zones["Vigorous"], y2=self.ecg.hr_max,
                         color='red', alpha=.3, label="Vigorous")

        ax1.set_ylabel("HR (bpm)")
        ax1.legend()
        ax1.set_ylim(40, self.ecg.hr_max)

        # LEFT WRIST DATA ============================================================================================
        ax2.set_title("Left Wrist")
        ax2.plot(self.df_epoch["Timestamp"], self.df_epoch["LWrist"], color='dodgerblue')
        ax2.set_ylabel("Counts / {} seconds".format(self.epoch_len))

        ax2.fill_between(x=self.df_epoch["Timestamp"],
                         y1=0,
                         y2=self.accel.cutpoint_dict["NonDomLight"],
                         color='grey', alpha=.3, label="Sedentary")

        ax2.fill_between(x=self.df_epoch["Timestamp"],
                         y1=self.accel.cutpoint_dict["NonDomLight"],
                         y2=self.accel.cutpoint_dict["NonDomModerate"],
                         color='green', alpha=.3, label="Light")

        ax2.fill_between(x=self.df_epoch["Timestamp"],
                         y1=self.accel.cutpoint_dict["NonDomModerate"],
                         y2=self.accel.cutpoint_dict["NonDomVigorous"],
                         color='orange', alpha=.3, label="Moderate")

        ax2.fill_between(x=self.df_epoch["Timestamp"],
                         y1=self.accel.cutpoint_dict["NonDomVigorous"],
                         y2=max(self.df_epoch["LWrist"]) * 1.1,
                         color='red', alpha=.3, label="Vigorous")
        ax2.legend()

        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
        ax2.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        plt.savefig("EpochedData_HRandLWrist.png")
        print("Plot saved as png ({})".format("EpochedHRandLWristData.png"))

    def calculate_activity_volume(self, start=None, stop=None, remove_invalid_ecg=True, show_plot=True):
        """Calculates activity volumes (minutes and percent of data) for LWrist and HR data. Able to crop.

           :arguments
           -start/stop: timestamps used to crop
           -remove_invalid_ecg: boolean whether to include invalid ECG signal periods.
                                If True, the total volume of data will be the same between LWrist and HR data.
                                If False, LWrist contains more data.
        """

        # Dataframe cropping =========================================================================================
        if start is not None and stop is None:
            stop = self.df_epoch.iloc[-1]["Timestamp"]

        if start is not None and stop is not None:
            df = self.df_epoch.loc[(self.df_epoch["Timestamp"] >= start) &
                                   (self.df_epoch["Timestamp"] <= stop)]

        if start is None and stop is None:
            df = self.df_epoch

        if remove_invalid_ecg:
            df = df.loc[df["HR Validity"] == "Valid"]

        print("\nCalculating activity data from {} to {} "
              "in {}-second epochs...".format(df.iloc[0]["Timestamp"], df.iloc[-1]['Timestamp'], self.epoch_len))

        # Non-dominant (left) wrist -----------------------------------------------------------------------------------
        nd_sed_epochs = df["LWrist"].loc[(df["LWrist"] < self.accel.cutpoint_dict["NonDomLight"])].shape[0]

        nd_light_epochs = df["LWrist"].loc[(df["LWrist"] >= self.accel.cutpoint_dict["NonDomLight"]) &
                                           (df["LWrist"] < self.accel.cutpoint_dict["NonDomModerate"])].shape[0]

        nd_mod_epochs = df["LWrist"].loc[(df["LWrist"] >= self.accel.cutpoint_dict["NonDomModerate"]) &
                                         (df["LWrist"] < self.accel.cutpoint_dict["NonDomVigorous"])].shape[0]

        nd_vig_epochs = df["LWrist"].loc[(df["LWrist"] >= self.accel.cutpoint_dict["NonDomVigorous"])].shape[0]

        # Heart rate -------------------------------------------------------------------------------------------------
        hr_sed_epochs = df.loc[df["HRR"] < 30].shape[0]
        hr_light_epochs = df.loc[(df["HRR"] >= 30) & (df["HRR"] < 40)].shape[0]
        hr_mod_epochs = df.loc[(df["HRR"] >= 40) & (df["HRR"] < 60)].shape[0]
        hr_vig_epochs = df.loc[df["HRR"] >= 60].shape[0]

        # Data storage ------------------------------------------------------------------------------------------------
        activity_minutes = {"LWristSed": round(nd_sed_epochs / (60 / self.epoch_len), 2),
                            "LWristLight": round(nd_light_epochs / (60 / self.epoch_len), 2),
                            "LWristMod": round(nd_mod_epochs / (60 / self.epoch_len), 2),
                            "LWristVig": round(nd_vig_epochs / (60 / self.epoch_len), 2),
                            "HRSed": round(hr_sed_epochs / (60 / self.epoch_len), 2),
                            "HRLight": round(hr_light_epochs / (60 / self.epoch_len), 2),
                            "HRMod": round(hr_mod_epochs / (60 / self.epoch_len), 2),
                            "HRVig": round(hr_vig_epochs / (60 / self.epoch_len), 2),
                            }

        activity_minutes["LWristMVPA"] = activity_minutes["LWristMod"] + activity_minutes["LWristVig"]
        activity_minutes["HRMVPA"] = activity_minutes["HRMod"] + activity_minutes["HRVig"]

        self.activity_volume = activity_minutes
        lwrist = [activity_minutes["LWristSed"], activity_minutes["LWristLight"],
                  activity_minutes["LWristMod"], activity_minutes["LWristVig"], activity_minutes["LWristMVPA"]]

        hr = [activity_minutes["HRSed"], activity_minutes["HRLight"],
              activity_minutes["HRMod"], activity_minutes["HRVig"], activity_minutes["HRMVPA"]]

        self.activity_df = pd.DataFrame(list(zip(lwrist, hr)), columns=["LWrist", "HR"])
        self.activity_df.index = ["Sedentary", "Light", "Moderate", "Vigorous", "MVPA"]

        self.activity_df["LWrist%"] = 100 * self.activity_df["LWrist"] / \
                                      sum(self.activity_df["LWrist"].loc[["Sedentary", "Light",
                                                                          "Moderate", "Vigorous"]])

        self.activity_df["LWrist%"] = self.activity_df["LWrist%"].round(2)

        self.activity_df["HR%"] = 100 * self.activity_df["HR"] / \
                                      sum(self.activity_df["HR"].loc[["Sedentary", "Light",
                                                                      "Moderate", "Vigorous"]])

        self.activity_df["HR%"] = self.activity_df["HR%"].round(2)

        print("\nActivity volume (removed invalid ECG epochs = {})".format(remove_invalid_ecg))
        print(self.activity_df)

        if show_plot:

            df = self.activity_df[["LWrist", "HR"]]

            plt.subplots(2, 2, figsize=(self.fig_width, self.fig_height))
            plt.suptitle("Comparison between LWrist and HR activity volumes")
            plt.subplots_adjust(hspace=.3)

            plt.subplot(2, 2, 1)
            plt.bar(["LWrist", "HR"], df.loc["Sedentary"], color=["dodgerblue", "red"], alpha=.75, edgecolor='black')
            plt.title("Sedentary")
            plt.ylabel("Minutes")

            plt.subplot(2, 2, 2)
            plt.bar(["LWrist", "HR"], df.loc["Light"], color=["dodgerblue", "red"], alpha=.75, edgecolor='black')
            plt.title("Light")

            plt.subplot(2, 2, 3)
            plt.bar(["LWrist", "HR"], df.loc["Moderate"], color=["dodgerblue", "red"], alpha=.75, edgecolor='black')
            plt.ylabel("Minutes")
            plt.title("Moderate")

            plt.subplot(2, 2, 4)
            plt.bar(["LWrist", "HR"], df.loc["Vigorous"], color=["dodgerblue", "red"], alpha=.75, edgecolor='black')
            plt.title("Vigorous")

            if start is None:
                start = self.df_epoch.iloc[0]["Timestamp"]
            if stop is None:
                stop = self.df_epoch.iloc[-1]["Timestamp"]

            start_format = datetime.strftime(datetime.strptime(str(start), "%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H_%M_%S")
            stop_format = datetime.strftime(datetime.strptime(str(stop), "%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H_%M_%S")

            f_name = self.check_file_overwrite("ActivityVolumes_InvalidECGRemoved{}_{}to{}".
                                               format(remove_invalid_ecg, start_format, stop_format))

            plt.savefig(f_name)
            print("Plot saved as png ({})".format(f_name))

    def plot_ecg_validity_data(self):
        """Calculates activity counts for chest, LWrist, and LAnkle during invalid and valid ECG signal periods.
           Plots means ± SD as barplot.
        """

        m = self.df_epoch[["LAnkle", "LWrist", "Chest", "HR Validity"]].groupby("HR Validity").mean()
        sd = self.df_epoch[["LAnkle", "LWrist", "Chest", "HR Validity"]].groupby("HR Validity").std()

        plt.subplots(1, 3, figsize=(self.fig_width, self.fig_height))
        plt.suptitle("ECG Validity by Activity Counts")

        plt.subplot(1, 3, 1)
        plt.title("Chest (mean ± SD)")
        plt.bar(["Invalid ECG", "Valid ECG"], m["Chest"], edgecolor='black', color=['red', 'green'], alpha=.5,
                yerr=sd["Chest"], capsize=4)

        plt.ylabel("Activity Counts")

        plt.subplot(1, 3, 2)
        plt.title("LWrist (mean ± SD)")
        plt.bar(["Invalid ECG", "Valid ECG"], m["LWrist"], edgecolor='black', color=['red', 'green'], alpha=.5,
                yerr=sd["LWrist"], capsize=4)

        plt.subplot(1, 3, 3)
        plt.title("LAnkle (mean ± SD)")
        plt.bar(["Invalid ECG", "Valid ECG"], m["LAnkle"], edgecolor='black', color=['red', 'green'], alpha=.5,
                yerr=sd["LAnkle"], capsize=4)

        plt.savefig("ECG_ValidityData_ActivityCounts.png")

    def recalculate_hr_epochs(self, epoch_len=15, show_plot=True):

        print("\n" + "Recalculating HR into {}-second epochs...".format(epoch_len))

        validity_list = []  # window's validity (binary; 1 = invalid)
        epoch_hr = []  # window's HRs
        avg_voltage = []  # window's voltage range

        raw = [i for i in self.ecg.df_raw["Raw"]]

        for start_index in range(0, len(raw) - epoch_len * self.ecg.sample_rate, epoch_len * self.ecg.sample_rate):

            qc = CheckQuality(ecg_object=self.ecg, start_index=start_index, epoch_len=epoch_len)

            avg_voltage.append(qc.volt_range)

            if qc.valid_period:
                validity_list.append("Valid")
                epoch_hr.append(round(qc.hr, 2))

            if not qc.valid_period:
                validity_list.append("Invalid")
                epoch_hr.append(0)

        print("Data has been re-epoched.")

        # List of epoched heart rates but any invalid epoch is marked as None instead of 0 (as is self.epoch_hr)
        valid_hr = [epoch_hr[i] if validity_list[i] == "Valid"
                    else None for i in range(len(epoch_hr))]

        df_epoch = pd.DataFrame(list(zip(self.ecg.df_raw["Timestamp"].iloc[::self.ecg.sample_rate * epoch_len],
                                         validity_list, epoch_hr, valid_hr)),
                                columns=["Timestamp", "Validity", "HR", "Valid HR"])

        self.hr_epoch = df_epoch

        if show_plot:

            fig, (ax1) = plt.subplots(1, sharex='col', figsize=(self.fig_width, self.fig_height))
            plt.title("Comparison of HR with different epoch lengths")
            ax1.plot(s.df_epoch["Timestamp"], s.df_epoch["Valid HR"],
                     color='red', label="Epoch{}".format(self.epoch_len))

            ax1.plot(s.hr_epoch["Timestamp"], s.hr_epoch["Valid HR"],
                     color='black', label="Epoch{}".format(epoch_len))

            ax1.set_ylabel("HR (bpm)")
            ax1.legend()

            xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
            ax1.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45, fontsize=8)

            plt.savefig("HR_EpochComparison_{}_and_{}_seconds.png".format(self.epoch_len, epoch_len))

    def plot_activity_counts(self):

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(self.fig_width, self.fig_height))
        plt.subplots_adjust(hspace=.30)
        plt.suptitle("Activity Count Comparison ({}-second epochs)".format(self.epoch_len))

        ax1.set_title("Chest")
        ax1.plot(self.df_epoch["Timestamp"], self.df_epoch["Chest"], color='green')
        ax1.set_ylabel("Counts")

        ax2.plot(self.df_epoch["Timestamp"], self.df_epoch["LWrist"], color='dodgerblue')
        ax2.set_title("LWrist")
        ax2.set_ylabel("Counts")

        ax3.plot(self.df_epoch["Timestamp"], self.df_epoch["LAnkle"], color='purple')
        ax3.set_title("LAnkle")
        ax3.set_ylabel("Counts")

        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
        ax1.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        plt.savefig("EpochedData_AllActivityCounts.png")

    def plot_ecg_validity(self, start=None, stop=None):
        """Plots raw ECG data and shades regions of invalid ECG signal. Able to crop by timestamp."""

        if start is None and stop is None:
            df = self.df_epoch[["Timestamp", "HR Validity"]].copy()
            df_ecg = self.ecg.df_raw[["Timestamp", "Raw"]]

        if start is not None and stop is not None:
            df = self.df_epoch.loc[(self.df_epoch["Timestamp"] >= start) &
                                   (self.df_epoch["Timestamp"] <= stop)][["Timestamp", "HR Validity"]]
            df_ecg = self.ecg.df_raw.loc[(self.ecg.df_raw["Timestamp"] >= start) &
                                         (self.ecg.df_raw["Timestamp"] <= stop)][["Timestamp", "Raw"]]

        df["HR Validity"] = [-32768 if i == "Valid" else 32768 for i in df["HR Validity"]]

        fig, ax1 = plt.subplots(1, figsize=(self.fig_width, self.fig_height))
        plt.suptitle("Raw ECG with invalid periods shaded")

        ax1.plot(df_ecg["Timestamp"], df_ecg["Raw"], color='red')
        ax1.set_ylabel("Voltage")

        ax1.fill_between(df["Timestamp"], -32768, df["HR Validity"], color='grey', alpha=.5)

        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
        ax1.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        if start is None:
            start = self.df_epoch.iloc[0]["Timestamp"]
        if stop is None:
            stop = self.df_epoch.iloc[-1]["Timestamp"]

        start_format = datetime.strftime(datetime.strptime(str(start), "%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H_%M_%S")
        stop_format = datetime.strftime(datetime.strptime(str(stop), "%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H_%M_%S")

        f_name = self.check_file_overwrite("ECG_SignalValidity_{}to{}".
                                           format(start_format, stop_format))

        plt.savefig(f_name)
        print("Plot saved as png ({})".format(f_name))


s = Subject(ecg_filepath="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 8/Anton_ECG.edf",
            ecg_downsample_ratio=2, epoch_len=15,
            leftwrist_filepath="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 8/Epoch_1s_LWrist.csv",
            leftankle_filepath="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 8/Epoch_1s_LAnkle.csv")

# Calculates activity volume using HR and scaled LWrist cutpoints in specified time period
# s.calculate_activity_volume(remove_invalid_ecg=True, start=None, stop=None, show_plot=True)

# Plots HR and LWrist activity count data with intensity ranges shaded
# s.plot_hr_wrist()

# Plots chest, LWrist, and LAnkle activity counts
# s.plot_activity_counts()

# Re-calculates HR averaged using new epoch lengths
# s.recalculate_hr_epochs(epoch_len=45, show_plot=True)

# Plot raw ECG data with invalid periods marked
# s.plot_ecg_validity(start=None, stop=None)

# Plots means ± SD of activity counts in valid and invalid ECG signal epochs
# s.plot_ecg_validity_data()
