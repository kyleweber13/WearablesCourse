import pyedflib  # toolbox for reading / writing EDF/BDF files
import numpy as np  # package for scientific computing
import pandas as pd  # package for data analysis and manipulation tools
from datetime import datetime  # module supplying classes for manipulating dates and times
from datetime import timedelta
import matplotlib.pyplot as plt  # library for creating static, animated, and interactive visualizations
import matplotlib.dates as mdates  # library for formatting plot axes labels as dates
import os  # module allowing code to use operating system dependent functionality
from scipy.signal import butter, filtfilt  # signal processing toolbox
import peakutils
import random


class Wearables:

    def __init__(self, leftwrist_filepath=None, leftankle_filepath=None, rightankle_filepath=None,
                 fig_height=7, fig_width=12):
        """Class that reads in GENEActiv data.
            Data is read in and no further methods are called.
            :arguments:
            -hip_filepath, ankle_filepath:
                    full pathway to all .edf files to read in. Default value is None;
                    fill will not be read in if no argument given
             -fig_heigth, fig_width: figure height and width in inches. Must be whole number.
            """

        # Default values for objects ----------------------------------------------------------------------------------
        self.lankle_fname = leftankle_filepath
        self.rankle_fname = rightankle_filepath
        self.wrist_fname = leftwrist_filepath

        self.fig_height = fig_height
        self.fig_width = fig_width

        self.accel_filter_low_f = None
        self.accel_filter_low_h = None

        self.df_epoched = None

        self.filter_run = False

        self.rankle_inverted = False  # boolean of whether right ankle x-axis has been inverted

        self.peaks_array = None
        self.peaks_axis = "Y"
        self.peaks_threshold_type = "normalized"
        self.peaks_thresh = 0.7
        self.peaks_min_dist = 250
        self.ankle_peaks = None  # data for all ankle peaks (right and left combined)

        # Methods and objects that are run automatically when class instance is created -------------------------------

        self.df_lankle, self.lankle_samplerate = self.load_correct_file(filepath=self.lankle_fname,
                                                                      f_type="Left Ankle")
        self.df_rankle, self.rankle_samplerate = self.load_correct_file(filepath=self.rankle_fname,
                                                                      f_type="Right Ankle")

        # Inverts x-axis data so it matches left ankle
        self.df_rankle["X"] = -self.df_rankle["X"]

        self.df_wrist, self.wrist_samplerate = self.load_correct_file(filepath=self.wrist_fname,
                                                                      f_type="Left Wrist")

        # 'Memory' stamps for previously-graphed region
        self.start_stamp = None
        self.stop_stamp = None

    def load_correct_file(self, filepath, f_type) -> object:
        """Method that specifies the correct file (.edf vs. .csv) to import for accelerometer files and
           retrieves sample rates.
        """

        if filepath is None:
            print("\nNo {} filepath given.".format(f_type))
            return None, None

        if ".csv" in filepath or ".CSV" in filepath:
            df, sample_rate = self.import_csv(filepath, f_type=f_type)

        if ".edf" in filepath or ".EDF" in filepath:
            df, sample_rate = self.import_edf(filepath, f_type=f_type)

        return df, sample_rate

    @staticmethod
    def import_edf(filepath=None, f_type="Accelerometer"):
        """Method that imports EDF from specified filepath and returns a df of timestamps relevant data.
           Works for both GENEACtiv EDF files.
           Also returns sampling rate.
           If no file was specified or a None was given, method returns None, None"""

        t0 = datetime.now()

        if filepath is not None and f_type != "ECG":
            print("------------------------------------- {} data -------------------------------------".format(f_type))
            print("\nImporting {}...".format(filepath))

            # READS IN ACCELEROMETER DATA ============================================================================
            file = pyedflib.EdfReader(filepath)

            x = file.readSignal(chn=0)
            y = file.readSignal(chn=1)
            z = file.readSignal(chn=2)

            vecmag = abs(np.sqrt(np.square(np.array([x, y, z])).sum(axis=0)) - 1)

            sample_rate = file.getSampleFrequencies()[1]  # sample rate
            starttime = file.getStartdatetime()

            file.close()

            # TIMESTAMP GENERATION ===================================================================================
            print("Creating timestamps...")

            end_time = starttime + timedelta(seconds=len(x) / sample_rate)
            timestamps = np.asarray(pd.date_range(start=starttime, end=end_time, periods=len(x)))
            print("Timestamps created.")

            print("\nAssembling data...")
            df = pd.DataFrame(list(zip(timestamps, x, y, z, vecmag)), columns=["Timestamp", "X", "Y", "Z", "Mag"])
            print("Dataframe created.")

            t1 = datetime.now()
            proc_time = (t1 - t0).seconds
            print("Import complete ({} seconds).".format(round(proc_time, 2)))

            return df, sample_rate

            # TIMESTAMP GENERATION ===================================================================================
        if filepath is not None and f_type == "ECG" or f_type == "ecg":
            print("Creating timestamps...")

            end_time = starttime + timedelta(seconds=len(ecg) / sample_rate)
            timestamps = np.asarray(pd.date_range(start=starttime, end=end_time, periods=len(ecg)))
            print("Timestamps created.")

            print("\nAssembling data...")
            df = pd.DataFrame(list(zip(timestamps, ecg)), columns=["Timestamp", "Raw"])
            print("Dataframe created.")

            t1 = datetime.now()
            proc_time = (t1 - t0).seconds
            print("Import complete ({} seconds).".format(round(proc_time, 2)))

            return df, sample_rate

        if filepath is None:
            print("------------------------------------- {} data -------------------------------------".format(f_type))
            print("No {} filepath given.".format(f_type))
            return None, None

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
        df.columns = ["Timestamp", "X", "Y", "Z", "Lux", "Button", "Temperature"]
        df = df[["Timestamp", "X", "Y", "Z"]]

        df["Timestamp"] = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S:%f") for i in df["Timestamp"]]

        vecmag = abs(np.sqrt(np.square(np.array([df["X"], df["Y"], df["Z"]])).sum(axis=0)) - 1)

        df["Mag"] = vecmag

        t1 = datetime.now()
        proc_time = (t1 - t0).seconds
        print("Import complete ({} seconds).".format(round(proc_time, 2)))

        return df, sample_rate

    def filter_signal(self, device_type="accelerometer", type="bandpass", low_f=1, high_f=10, filter_order=1):
        """Filtering details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
        Arguments:
            -device_type: "accelerometer"
            -type: filter type - "bandpass", "lowpass", or "highpass"
            -low_f: low-end cutoff frequency, required for lowpass and bandpass filters
            -high_f: high-end cutoff frequency, required for highpass and bandpass filters
            -filter_order: integet for filter order
        Adds columns to dataframe corresponding to each device. Filters all devices that are available.
        """

        if device_type == "accelerometer":

            self.accel_filter_low_f = low_f
            self.accel_filter_low_h = high_f

            for data_type in ["lankle", "rankle", "lwrist"]:

                # DATA SET UP

                if data_type == "lankle" and self.lankle_fname is not None:
                    data = np.array([self.df_lankle["X"], self.df_lankle["Y"],
                                     self.df_lankle["Z"], self.df_lankle["Mag"]])
                    original_df = self.df_lankle
                    fs = self.lankle_samplerate * .5

                if data_type == "rankle" and self.rankle_fname is not None:
                    data = np.array([self.df_rankle["X"], self.df_rankle["Y"],
                                     self.df_rankle["Z"], self.df_rankle["Mag"]])
                    original_df = self.df_rankle
                    fs = self.rankle_samplerate * .5

                if data_type == "lwrist" and self.wrist_fname is not None:
                    data = np.array([self.df_wrist["X"], self.df_wrist["Y"],
                                     self.df_wrist["Z"], self.df_wrist["Mag"]])
                    original_df = self.df_wrist
                    fs = self.wrist_samplerate * .5

                # FILTERING TYPES
                if type == "lowpass":
                    print("\nFiltering {} accelerometer data with {}Hz, "
                          "order {} lowpass filter.".format(data_type, low_f, filter_order))
                    low = low_f / fs
                    b, a = butter(N=filter_order, Wn=low, btype="lowpass")
                    filtered_data = filtfilt(b, a, x=data)

                if type == "highpass":
                    print("\nFiltering {} accelerometer data with {}Hz, "
                          "order {} highpass filter.".format(data_type, high_f, filter_order))
                    high = high_f / fs
                    b, a = butter(N=filter_order, Wn=high, btype="highpass")
                    filtered_data = filtfilt(b, a, x=data)

                if type == "bandpass":
                    print("\nFiltering {} accelerometer data with {}-{}Hz, "
                          "order {} bandpass filter.".format(data_type, low_f, high_f, filter_order))

                    low = low_f / fs
                    high = high_f / fs
                    b, a = butter(N=filter_order, Wn=[low, high], btype="bandpass")
                    filtered_data = filtfilt(b, a, x=data)

                original_df["X_filt"] = filtered_data[0]
                original_df["Y_filt"] = filtered_data[1]
                original_df["Z_filt"] = filtered_data[2]
                original_df["Mag_filt"] = np.abs(filtered_data[3])

        print("\nFiltering complete.")
        self.filter_run = True

    def find_peaks(self, min_dist_ms=250, thresh_type="normalized", threshold=0.7, axis="y"):
        """Runs peak detection algorithm according to specified arguments. Keeps track of last set of parameters
           used with self.peak_axis, self.peaks_threshold_type, self.peaks_threshold and self.peaks_min_dist.
           Requires data to have been filtered.

           Outputs self.peaks_array: peaks_array[0] is LAnkle, [1] is RAnkle, and [2] is LWrist

        :argument
        -min_dist_ms: minimum distance required between peaks in milliseconds. Function converts to datapoints.
        -thresh_type: type of threshold to use; "normalized" or "absolute"
        -threshold: value corresponding to threshold. If thresh_type == "normalized", threshold needs to be
        between 0 and 1; this represents the fraction of the signal range to use as the threshold. If thresh_type ==
        "absolute", threshold can be any value that corresponds to the threshold in G's
        -axis: which accelerometer data to use; "x", "y", "z", or "mag" (vector magnitude)
        -show_plot: boolean to plot data or not
        """

        print("\nDetecting peaks using the following parameters:")
        print("-Detecting peaks using {} data".format(axis.capitalize()))
        print("-Minimum distance between peaks = {} ms".format(min_dist_ms))
        print("-Threshold type = {}".format(thresh_type))
        if thresh_type == "normalized":
            print("-Threshold = {}% of signal amplitude".format(threshold*100))
        if thresh_type == "absolute":
            print("-Threshod = {} G".format(threshold))

        if not self.filter_run:
            print("\nPlease run filter and try again.")
            return None

        lankle_peaks = peakutils.indexes(y=self.df_lankle["{}_filt".format(axis.capitalize())],
                                         min_dist=int(self.lankle_samplerate/(1000/min_dist_ms)),
                                         thres_abs=False if thresh_type == "normalized" else True, thres=threshold)

        rankle_peaks = peakutils.indexes(y=self.df_rankle["{}_filt".format(axis.capitalize())],
                                         min_dist=int(self.rankle_samplerate/(1000/min_dist_ms)),
                                         thres_abs=False if thresh_type == "normalized" else True, thres=threshold)

        wrist_peaks = peakutils.indexes(y=self.df_wrist["{}_filt".format(axis.capitalize())],
                                        min_dist=int(self.wrist_samplerate/(1000/min_dist_ms)),
                                        thres_abs=False if thresh_type == "normalized" else True, thres=threshold)

        print("\nPeak detection results:")
        print("-Left ankle: {} peaks".format(len(lankle_peaks)))
        print("-Right ankle: {} peaks".format(len(rankle_peaks)))
        print("-Left wrist: {} peaks".format(len(wrist_peaks)))

        self.peaks_array = np.array([lankle_peaks, rankle_peaks, wrist_peaks], dtype='object')
        self.peaks_axis = axis
        self.peaks_threshold_type = thresh_type
        self.peaks_thresh = threshold
        self.peaks_min_dist = min_dist_ms

        self.ankle_peaks = sorted(np.concatenate((self.peaks_array[0], self.peaks_array[1])))

    def get_timestamps(self, start=None, stop=None):

        # If start/stop given as integer, sets start/stop stamps to minutes into collection ---------------------------
        if (type(start) is int or type(start) is float) and (type(stop) is int or type(stop) is float):

            data_type = "absolute"

            start_stamp = self.df_lankle["Timestamp"].iloc[0] + timedelta(minutes=start)
            stop_stamp = self.df_lankle["Timestamp"].iloc[0] + timedelta(minutes=stop)

        # Formats arguments as datetimes -----------------------------------------------------------------------------

        # If arguments are given and no previous region has been specified
        else:

            data_type = "timestamp"

            if start is not None and self.start_stamp is None:
                try:
                    start_stamp = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    start_stamp = datetime.strptime(start, "%Y-%m-%d %H:%M:%S.%f")

            if stop is not None and self.stop_stamp is None:
                try:
                    stop_stamp = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    stop_stamp = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S.%f")

            # If arguments not given and no stamps from previous region
            # Sets start/stop to first/last timestamp for hip or ankle data
            if start is None and self.start_stamp is None:
                start_stamp = self.df_lankle["Timestamp"].iloc[0]

            if stop is None and self.stop_stamp is None:
                stop_stamp = self.df_lankle["Timestamp"].iloc[-1]

            # If arguments are not given and there are stamps from previous region
            if start is None and self.start_stamp is not None:
                print("Plotting previously-plotted region.")
                start_stamp = self.start_stamp
            if stop is None and self.stop_stamp is not None:
                stop_stamp = self.stop_stamp

            # If arguments given --> overrides stamps from previous region
            if start is not None and self.start_stamp is not None:
                try:
                    start_stamp = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    start_stamp = datetime.strptime(start, "%Y-%m-%d %H:%M:%S.%f")

            if stop is not None and self.stop_stamp is not None:
                try:
                    stop_stamp = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    stop_stamp = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S.%f")

        return start_stamp, stop_stamp, data_type

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

    @staticmethod
    def set_xaxis_format(window_len):

        # Formatting x-axis ticks ------------------------------------------------------------------------------------
        if window_len >= .25:
            xfmt = mdates.DateFormatter("%a %b %d, %H:%M:%S")
            bottom_plot_crop_value = .17
        # Shows milliseconds if plotting less than 15-second window
        if window_len < .25:
            xfmt = mdates.DateFormatter("%a %b %d, %H:%M:%S.%f")
            bottom_plot_crop_value = .23

        # Generates ~15 ticks (1/15th of window length apart)
        locator = mdates.MinuteLocator(byminute=np.arange(0, 59, int(np.ceil(window_len / 15))), interval=1)

        # Two-second ticks if window length between 5 and 30 seconds
        if 1 / 12 < window_len <= .5:
            locator = mdates.SecondLocator(interval=2)

        # Plots ~15 ticks if window less than 5 seconds
        if window_len <= 1 / 12:
            locator = mdates.MicrosecondLocator(interval=int(1000000 * (window_len * 60 / 15)))

        return xfmt, locator, bottom_plot_crop_value

    def plot_ankles_x(self, start=None, stop=None, downsample_factor=1, use_filtered=False):
        """Generates two subplots for x-axis accleration data.
           Top: original left and right ankle data. Bottom: original left ankle data and inverted right ankle data.

           Inverts x-axis in right ankle dataframe once complete. Keeps track of whether current inversion state is
           the original or inverted which occurs with multiple function calls."""

        # Plot set up ------------------------------------------------------------------------------------------------
        start_stamp, stop_stamp, data_type = self.get_timestamps(start, stop)
        window_len = (stop_stamp - start_stamp).seconds / 60
        xfmt, locator, bottom_plot_crop_value = self.set_xaxis_format(window_len=window_len)

        # Sets 'memory' values to current start/stop values
        self.start_stamp = start_stamp
        self.stop_stamp = stop_stamp

        # Sets xlim so legend shouldn't overlap any data
        buffer_len = window_len / 6
        x_lim = (self.start_stamp - timedelta(minutes=buffer_len), self.stop_stamp + timedelta(minutes=buffer_len/2))

        # Sets stop to end of collection if stop timestamp exceeds timestamp range
        try:
            if stop_stamp > self.df_lankle.iloc[-1]["Timestamp"]:
                stop_stamp = self.df_lankle.iloc[-1]["Timestamp"]
        except TypeError:
            if datetime.strptime(stop_stamp, "%Y-%m-%d %H:%M:%S") > self.df_lankle.iloc[-1]["Timestamp"]:
                stop_stamp = self.df_lankle.iloc[-1]["Timestamp"]

        df_lankle = self.df_lankle.loc[(self.df_lankle["Timestamp"] > start_stamp) &
                                       (self.df_lankle["Timestamp"] < stop_stamp)]
        df_rankle = self.df_rankle.loc[(self.df_rankle["Timestamp"] > start_stamp) &
                                       (self.df_rankle["Timestamp"] < stop_stamp)]

        if data_type == "absolute":
            df_lankle["Timestamp"] = np.arange(0, (stop_stamp - start_stamp).seconds,
                                               1 / self.df_lankle)[0:df_lankle.shape[0]]
            df_rankle["Timestamp"] = np.arange(0, (stop_stamp - start_stamp).seconds,
                                               1 / self.df_rankle)[0:df_rankle.shape[0]]

        if downsample_factor != 1:
            df_lankle = df_lankle.iloc[::downsample_factor, :]
            df_rankle = df_rankle.iloc[::downsample_factor, :]

        # Plotting ---------------------------------------------------------------------------------------------------
        if use_filtered:
            col_name_suff = "_filt"
        if not use_filtered:
            col_name_suff = ""

        fig, (ax1, ax2) = plt.subplots(2, sharex="col", figsize=(self.fig_width, self.fig_height))
        plt.subplots_adjust(bottom=bottom_plot_crop_value)

        ax1.set_title("Original Data (filtered = {})".format(use_filtered))
        ax1.plot(df_lankle["Timestamp"], df_lankle["X" + col_name_suff], color='black', label="LAnkle_X")

        ax1.plot(df_rankle["Timestamp"], -df_rankle["X" + col_name_suff], color='red', label="RAnkle_X")
        ax1.set_xlim(x_lim)
        ax1.set_ylabel("G")
        ax1.legend(loc='upper left')

        ax2.set_title("Right Ankle Inverted X-Axis")
        ax2.plot(df_lankle["Timestamp"], df_lankle["X" + col_name_suff], color='black', label="LAnkle_X")
        ax2.plot(df_rankle["Timestamp"], df_rankle["X" + col_name_suff], color='red', label="Negative RAnkle_X")
        ax2.set_xlim(x_lim)
        ax2.set_ylabel("G")
        ax2.legend(loc='upper left')

        ax2.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=8)

    def plot_ankle_data(self, start=None, stop=None, downsample_factor=1, use_filtered=True):
        """Generates plot of right and left ankle data. One subplot for each axis (x, y, z) with both ankles
           plotted on same subplot.

           Able to set start/stop times using start/stop arguments. Downsample using downsample_factor
        """

        # Plot set up ------------------------------------------------------------------------------------------------
        if use_filtered:
            col_name_suff = "_filt"
        if not use_filtered:
            col_name_suff = ""

        start_stamp, stop_stamp, data_type = self.get_timestamps(start, stop)
        window_len = (stop_stamp - start_stamp).seconds / 60
        xfmt, locator, bottom_plot_crop_value = self.set_xaxis_format(window_len=window_len)

        # Sets 'memory' values to current start/stop values
        self.start_stamp = start_stamp
        self.stop_stamp = stop_stamp

        # Sets xlim so legend shouldn't overlap any data
        buffer_len = window_len / 8
        x_lim = (self.start_stamp - timedelta(minutes=buffer_len), self.stop_stamp + timedelta(minutes=buffer_len/2))

        # Sets stop to end of collection if stop timestamp exceeds timestamp range
        try:
            if stop_stamp > self.df_lankle.iloc[-1]["Timestamp"]:
                stop_stamp = self.df_lankle.iloc[-1]["Timestamp"]
        except TypeError:
            if datetime.strptime(stop_stamp, "%Y-%m-%d %H:%M:%S") > self.df_lankle.iloc[-1]["Timestamp"]:
                stop_stamp = self.df_lankle.iloc[-1]["Timestamp"]

        df_lankle = self.df_lankle.loc[(self.df_lankle["Timestamp"] > start_stamp) &
                                       (self.df_lankle["Timestamp"] < stop_stamp)]
        df_rankle = self.df_rankle.loc[(self.df_rankle["Timestamp"] > start_stamp) &
                                       (self.df_rankle["Timestamp"] < stop_stamp)]

        if data_type == "absolute":
            df_lankle["Timestamp"] = np.arange(0, (stop_stamp - start_stamp).seconds,
                                               1 / self.df_lankle)[0:df_lankle.shape[0]]
            df_rankle["Timestamp"] = np.arange(0, (stop_stamp - start_stamp).seconds,
                                               1 / self.df_rankle)[0:df_rankle.shape[0]]
        if downsample_factor != 1:
            df_lankle = df_lankle.iloc[::downsample_factor, :]
            df_rankle = df_rankle.iloc[::downsample_factor, :]

        # Plotting ---------------------------------------------------------------------------------------------------

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex="col", sharey='col', figsize=(self.fig_width, self.fig_height))
        plt.suptitle("All Accelerometer Data (filtered = {})".format(use_filtered))

        plt.subplots_adjust(bottom=bottom_plot_crop_value)
        ax1.plot(df_lankle["Timestamp"], df_lankle["X" + col_name_suff], color='black', label="LA_x")
        ax1.plot(df_rankle["Timestamp"], df_rankle["X" + col_name_suff], color='red', label="RA_x")
        ax1.set_ylabel("G")
        ax1.legend(loc='upper left')
        ax1.set_xlim(x_lim)

        ax2.plot(df_lankle["Timestamp"], df_lankle["Y" + col_name_suff], color='black', label="LA_y")
        ax2.plot(df_rankle["Timestamp"], df_rankle["Y" + col_name_suff], color='red', label="RA_y")
        ax2.set_ylabel("G")
        ax2.legend(loc='upper left')
        ax2.set_xlim(x_lim)

        ax3.plot(df_lankle["Timestamp"], df_lankle["Z" + col_name_suff], color='black', label="LA_z")
        ax3.plot(df_rankle["Timestamp"], df_rankle["Z" + col_name_suff], color='red', label="RA_z")
        ax3.set_ylabel("G")
        ax3.legend(loc='upper left')
        ax3.set_xlim(x_lim)

        ax3.xaxis.set_major_formatter(xfmt)
        ax3.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=8)

    def plot_all_data(self, start=None, stop=None, downsample_factor=1, axis="x"):
        """Plots both ankles on one subplot and wrist on the second subplot.
        Able to specify axis using 'axis' argument ("x", "y", "z", "mag")
        """

        if not self.filter_run:
            print("\nPlease filter data and try again.")
            return None

        # Plot set up ------------------------------------------------------------------------------------------------
        start_stamp, stop_stamp, data_type = self.get_timestamps(start, stop)
        window_len = (stop_stamp - start_stamp).seconds / 60
        xfmt, locator, bottom_plot_crop_value = self.set_xaxis_format(window_len=window_len)

        # Sets 'memory' values to current start/stop values
        self.start_stamp = start_stamp
        self.stop_stamp = stop_stamp

        # Sets xlim so legend shouldn't overlap any data
        buffer_len = window_len / 8
        x_lim = (self.start_stamp - timedelta(minutes=buffer_len), self.stop_stamp + timedelta(minutes=buffer_len/2))

        # Sets stop to end of collection if stop timestamp exceeds timestamp range
        try:
            if stop_stamp > self.df_lankle.iloc[-1]["Timestamp"]:
                stop_stamp = self.df_lankle.iloc[-1]["Timestamp"]
        except TypeError:
            if datetime.strptime(stop_stamp, "%Y-%m-%d %H:%M:%S") > self.df_lankle.iloc[-1]["Timestamp"]:
                stop_stamp = self.df_lankle.iloc[-1]["Timestamp"]

        df_lankle = self.df_lankle.loc[(self.df_lankle["Timestamp"] > start_stamp) &
                                       (self.df_lankle["Timestamp"] < stop_stamp)]
        df_rankle = self.df_rankle.loc[(self.df_rankle["Timestamp"] > start_stamp) &
                                       (self.df_rankle["Timestamp"] < stop_stamp)]
        df_wrist = self.df_wrist.loc[(self.df_wrist["Timestamp"] > start_stamp) &
                                       (self.df_wrist["Timestamp"] < stop_stamp)]

        if data_type == "absolute":
            df_lankle["Timestamp"] = np.arange(0, (stop_stamp - start_stamp).seconds,
                                               1 / self.df_lankle)[0:df_lankle.shape[0]]
            df_rankle["Timestamp"] = np.arange(0, (stop_stamp - start_stamp).seconds,
                                               1 / self.df_rankle)[0:df_rankle.shape[0]]
            df_wrist["Timestamp"] = np.arange(0, (stop_stamp - start_stamp).seconds,
                                              1 / self.df_wrist)[0:df_wrist.shape[0]]

        if downsample_factor != 1:
            df_lankle = df_lankle.iloc[::downsample_factor, :]
            df_rankle = df_rankle.iloc[::downsample_factor, :]
            df_wrist = df_wrist.iloc[::downsample_factor, :]

        # Plotting ----------------------------------------------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(2, sharex='col', sharey='col', figsize=(self.fig_width, self.fig_height))
        plt.subplots_adjust(bottom=bottom_plot_crop_value)

        ax1.plot(df_lankle["Timestamp"], df_lankle["{}_filt".format(axis.capitalize())],
                 color='black', label="LA_{}".format(axis.capitalize()))
        ax1.plot(df_rankle["Timestamp"], df_rankle["{}_filt".format(axis.capitalize())],
                 color='red', label="RA_{}".format(axis.capitalize()))
        ax1.set_ylabel("G")
        ax1.legend(loc='upper left')
        ax1.set_xlim(x_lim)

        ax2.plot(df_wrist["Timestamp"], df_wrist["{}_filt".format(axis.capitalize())],
                 color="blue", label="Wrist_{}".format(axis.capitalize()))
        ax2.set_ylabel("G")
        ax2.legend(loc='upper left')
        ax2.set_xlim(x_lim)

        ax2.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=8)

    def plot_detected_peaks(self, start=None, stop=None):
        """Plots detected peaks from self.find_peaks() marked on whichever accelerometer axis was used.
        LAnkle, RAnkle, and LWrist data on separate subplots.
        """

        if not self.filter_run:
            print("\nPlease filter data and try again.")
            return None

        if self.peaks_array is None:
            print("\nNo peaks have been detected. Run x.find_peaks() first and then try again.")
            return None

        # Plot set up ------------------------------------------------------------------------------------------------
        start_stamp, stop_stamp, data_type = self.get_timestamps(start, stop)
        window_len = (stop_stamp - start_stamp).seconds / 60
        xfmt, locator, bottom_plot_crop_value = self.set_xaxis_format(window_len=window_len)

        # Sets 'memory' values to current start/stop values
        self.start_stamp = start_stamp
        self.stop_stamp = stop_stamp

        # Sets xlim so legend shouldn't overlap any data
        buffer_len = window_len / 8
        x_lim = (self.start_stamp - timedelta(minutes=buffer_len), self.stop_stamp + timedelta(minutes=1))

        # Sets stop to end of collection if stop timestamp exceeds timestamp range
        try:
            if stop_stamp > self.df_lankle.iloc[-1]["Timestamp"]:
                stop_stamp = self.df_lankle.iloc[-1]["Timestamp"]
        except TypeError:
            if datetime.strptime(stop_stamp, "%Y-%m-%d %H:%M:%S") > self.df_lankle.iloc[-1]["Timestamp"]:
                stop_stamp = self.df_lankle.iloc[-1]["Timestamp"]

        df_lankle = self.df_lankle.loc[(self.df_lankle["Timestamp"] > start_stamp) &
                                       (self.df_lankle["Timestamp"] < stop_stamp)]
        df_rankle = self.df_rankle.loc[(self.df_rankle["Timestamp"] > start_stamp) &
                                       (self.df_rankle["Timestamp"] < stop_stamp)]
        df_wrist = self.df_wrist.loc[(self.df_wrist["Timestamp"] > start_stamp) &
                                       (self.df_wrist["Timestamp"] < stop_stamp)]

        # Finds which datapoint corresponds to start of cropped df. Needed to adjust peak indexes.
        start_offset = self.df_lankle.loc[self.df_lankle["Timestamp"] >= start_stamp].index[0]
        end_offset = self.df_lankle.loc[self.df_lankle["Timestamp"] >= stop_stamp].index[0]

        if data_type == "absolute":
            df_lankle["Timestamp"] = np.arange(0, (stop_stamp - start_stamp).seconds,
                                               1 / self.df_lankle)[0:df_lankle.shape[0]]
            df_rankle["Timestamp"] = np.arange(0, (stop_stamp - start_stamp).seconds,
                                               1 / self.df_rankle)[0:df_rankle.shape[0]]
            df_wrist["Timestamp"] = np.arange(0, (stop_stamp - start_stamp).seconds,
                                              1 / self.df_wrist)[0:df_wrist.shape[0]]

        # Plotting ----------------------------------------------------------------------------------------------------
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex="col", figsize=(self.fig_width, self.fig_height))
        plt.subplots_adjust(bottom=bottom_plot_crop_value)

        plt.suptitle("{} data ({} threshold of {}, min_dist = {}ms)".format(self.peaks_axis.capitalize(),
                                                                            self.peaks_threshold_type,
                                                                            self.peaks_thresh, self.peaks_min_dist))

        # Left Ankle -------------------------------------------------------------------------------------------------
        ax1.plot(df_lankle["Timestamp"], df_lankle["{}_filt".format(self.peaks_axis.capitalize())],
                 color='black', label="LA_{}".format(self.peaks_axis))
        if len(self.peaks_array[0]) > 0:
            ax1.plot([self.df_lankle["Timestamp"].iloc[i] for i in self.peaks_array[0] if
                      start_offset <= i <= end_offset],
                     [self.df_lankle["{}_filt".format(self.peaks_axis.capitalize())].iloc[i] for
                      i in self.peaks_array[0] if start_offset <= i <= end_offset],
                     linestyle="", color='red', marker="x")
        ax1.set_ylabel("G")
        ax1.legend(loc='upper left')
        ax1.set_xlim(x_lim)

        # Right Ankle -------------------------------------------------------------------------------------------------
        ax2.plot(df_rankle["Timestamp"], df_rankle["{}_filt".format(self.peaks_axis.capitalize())],
                 color='red', label="RA_{}".format(self.peaks_axis))

        if len(self.peaks_array[1]) > 0:
            ax2.plot([self.df_rankle["Timestamp"].iloc[i] for i in self.peaks_array[1] if
                      start_offset <= i <= end_offset],
                     [self.df_rankle["{}_filt".format(self.peaks_axis.capitalize())].iloc[i] for
                      i in self.peaks_array[1] if start_offset <= i <= end_offset],
                     linestyle="", color='black', marker="x")
        ax2.set_ylabel("G")
        ax2.legend(loc='upper left')
        ax2.set_xlim(x_lim)

        # Left Wrist --------------------------------------------------------------------------------------------------
        ax3.plot(df_wrist["Timestamp"], df_wrist["{}_filt".format(self.peaks_axis.capitalize())],
                 color='blue', label="LW_{}".format(self.peaks_axis))
        if len(self.peaks_array[2]) > 0:
            ax3.plot([self.df_wrist["Timestamp"].iloc[i] for i in self.peaks_array[2] if start_offset <= i <= end_offset],
                     [self.df_wrist["{}_filt".format(self.peaks_axis.capitalize())].iloc[i] for
                      i in self.peaks_array[2] if start_offset <= i <= end_offset],
                     linestyle="", color='black', marker="x")
        ax3.set_ylabel("G")
        ax3.legend(loc='upper left')
        ax3.set_xlim(x_lim)

        ax3.xaxis.set_major_formatter(xfmt)
        ax3.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=8)

        f_name = self.check_file_overwrite("DetectedPeaks_{}Thresh_{}Axis_{}ms_{} "
                                           "to {}".format(self.peaks_threshold_type.capitalize(),
                                                          self.peaks_axis.capitalize(), self.peaks_min_dist,
                                                          datetime.strftime(start_stamp,
                                                                            "%Y-%m-%d %H_%M_%S"),
                                                          datetime.strftime(stop_stamp,
                                                                            "%Y-%m-%d %H_%M_%S")))
        plt.savefig(f_name)
        print("Plot saved as png ({})".format(f_name))

    def compare_walk_and_run(self):

        pass


x = Wearables(leftankle_filepath="Data Files/Labs3and4_LAnkle.csv",
              rightankle_filepath="Data Files/Labs3and4_RAnkle.csv",
              leftwrist_filepath="Data Files/Labs3and4_LWrist_1.csv")
x.filter_signal(device_type="accelerometer", type="bandpass", low_f=0.5, high_f=10, filter_order=3)
# x.plot_ankles_x(use_filtered=False, start="2020-09-23 08:24:59.00", stop="2020-09-23 08:25:02.00")
# x.plot_ankle_data(use_filtered=True, start="2020-09-23 08:24:59.00", stop="2020-09-23 08:25:02.00")
# x.plot_all_data(axis="mag", start="2020-09-23 08:24:59.00", stop="2020-09-23 08:25:02.00")
# x.find_peaks(min_dist_ms=300, thresh_type="absolute", threshold=1.5, axis="x")
# x.plot_detected_peaks()
