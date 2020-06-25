import pyedflib
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from scipy.signal import butter, lfilter, filtfilt


class GENEActiv:

    def __init__(self, wrist_filepath=None, hip_filepath=None):
        """Class that reads in GENEActiv data in EDF format."""

        self.wrist_fname = wrist_filepath
        self.hip_fname = hip_filepath

        self.df_wrist, self.wrist_samplerate = self.load_correct_file(filepath=self.wrist_fname, f_type="wrist")
        self.df_hip, self.hip_samplerate = self.load_correct_file(filepath=self.hip_fname, f_type="hip")

        # 'Memory' stamps for previously-graphed region
        self.start_stamp = None
        self.stop_stamp = None

    def load_correct_file(self, filepath, f_type):

        if filepath is None:
            print("\nNo {} filepath given.".format(f_type))
            return None, None

        if ".csv" in filepath or ".CSV" in filepath:
            df, sample_rate = self.import_csv(filepath, f_type=f_type)

        if ".edf" in filepath or ".EDF" in filepath:
            df, sample_rate = self.import_edf(filepath, f_type=f_type)

        return df, sample_rate

    @staticmethod
    def import_edf(filepath, f_type):
        """Method that imports EDF from specified filepath and returns a df of timestamps and x, y, z channels.
           Also returns sampling rate.
           If no file was specified or a None was given, method returns None, None"""

        t0 = datetime.now()

        if filepath is not None:
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

        if filepath is None:
            print("------------------------------------- {} data -------------------------------------".format(f_type))
            print("No {} filepath given.".format(f_type))
            return None, None

    @staticmethod
    def import_csv(filepath, f_type):

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

    def filter_signal(self, data_type=None, type="bandpass", low_f=1, high_f=10, sample_f=75, filter_order=1):
        """Filtering details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html

        Arguments:
            -data_type: "hip" or "wrist"
            -type: filter type - "bandpass", "lowpass", or "highpass"
            -low_f: low-end cutoff frequency, required for lowpass and bandpass filters
            -high_f: high-end cutoff frequency, required for highpass and bandpass filters
            -sample_f: sample rate, Hz
            -filter_order: integet for filter order

        Adds columns to dataframe corresponding to "data_type" argument of filtered data
        """

        # Normalizes frequencies
        nyquist_freq = 0.5 * sample_f

        if data_type == "wrist" or data_type == "Wrist":
            data = np.array([self.df_wrist["X"], self.df_wrist["Y"], self.df_wrist["Z"]])
            original_df = self.df_wrist
        if data_type == "hip" or data_type == "Hip":
            data = np.array([self.df_hip["X"], self.df_hip["Y"], self.df_hip["Z"]])
            original_df = self.df_hip

        if type == "lowpass":
            print("\nFiltering {} accelerometer data with {}Hz, order {} lowpass filter.".format(data_type,
                                                                                                 low_f,
                                                                                                 filter_order))
            low = low_f / nyquist_freq
            b, a = butter(N=filter_order, Wn=low, btype="lowpass")
            filtered_data = filtfilt(b, a, x=data)

        if type == "highpass":
            print("\nFiltering {} accelerometer data with {}Hz, order {} highpass filter.".format(data_type,
                                                                                                  high_f,
                                                                                                  filter_order))
            high = high_f / nyquist_freq
            b, a = butter(N=filter_order, Wn=high, btype="highpass")
            filtered_data = filtfilt(b, a, x=data)

        if type == "bandpass":
            print("\nFiltering {} accelerometer data with {}-{}Hz, order {} bandpass filter.".format(data_type,
                                                                                                     low_f, high_f,
                                                                                                     filter_order))

            low = low_f / nyquist_freq
            high = high_f / nyquist_freq
            b, a = butter(N=filter_order, Wn=[low, high], btype="bandpass")
            filtered_data = filtfilt(b, a, x=data)

        original_df["X_filt"] = filtered_data[0]
        original_df["Y_filt"] = filtered_data[1]
        original_df["Z_filt"] = filtered_data[2]

    def plot_data(self, start=None, stop=None, downsample_factor=1):
        """Plots hip and wrist data whichever/both is available.

            arguments:
                -start: timestamp for start of region. Format = "YYYY-MM-DD HH:MM:SS"
                -stop: timestamp for end of region. Format = "YYYY-MM-DD HH:MM:SS"
                -downsample: ratio by which data are downsampled. E.g. downsample=3 will downsample from 75 to 25 Hz

            If start and stop are not specified, data will be cropped to one of the following:
                -If no previous graphs have been generated, it will plot the entire data file
                -If a previous crop has occurred, it will 'remember' that region and plot it again.

            To clear the 'memory' of previously-plotted regions, enter "x.start_stamp=None"
            and "x.stop_stop=None" in console
        """

        print("\n-----------------------------------------------------------------------------------------------------")

        # Formats arguments as datetimes ------------------------------------------------------------------------------

        # If arguments are given and no previous region has been specified
        if start is not None and self.start_stamp is None:
            start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
            self.start_stamp = start
        if stop is not None and self.stop_stamp is None:
            stop = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S")
            self.stop_stamp = stop

        # If arguments not given and no stamps from previous region
        if start is None and self.start_stamp is None:
            try:
                start = self.df_hip["Timestamp"][0]
            except (AttributeError, ValueError):
                start = self.df_wrist["Timestamp"][0]

        if stop is None and self.stop_stamp is None:
            try:
                stop = self.df_hip["Timestamp"].iloc[-1]
            except (AttributeError, ValueError):
                stop = self.df_wrist["Timestamp"].iloc[-1]

        # If are not given and there are stamps from previous region
        if start is None and self.start_stamp is not None:
            print("Plotting previously-plotted region.")
            start = self.start_stamp
        if stop is None and self.stop_stamp is not None:
            stop = self.stop_stamp

        # Crops dataframes to selected region -------------------------------------------------------------------------
        if self.hip_fname is not None:
            # Sets stop to end of collection if stop timestamp exceeds timestamp range
            try:
                if stop > self.df_hip.iloc[-1]["Timestamp"]:
                    stop = self.df_hip.iloc[-1]["Timestamp"]
            except TypeError:
                if datetime.strptime(stop, "%Y-%m-%d %H:%M:%S") > self.df_hip.iloc[-1]["Timestamp"]:
                    stop = self.df_hip.iloc[-1]["Timestamp"]

            df_hip = self.df_hip.loc[(self.df_hip["Timestamp"] > start) & (self.df_hip["Timestamp"] < stop)]

            if downsample_factor != 1:
                df_hip = df_hip.iloc[::downsample_factor, :]

        if self.wrist_fname is not None:
            # Sets stop to end of collection if stop timestamp exceeds timestamp range
            try:
                if stop > self.df_wrist.iloc[-1]["Timestamp"]:
                    stop = self.df_wrist.iloc[-1]["Timestamp"]
            except TypeError:
                if datetime.strptime(stop, "%Y-%m-%d %H:%M:%S") > self.df_wrist.iloc[-1]["Timestamp"]:
                    stop = self.df_wrist.iloc[-1]["Timestamp"]

            df_wrist = self.df_wrist.loc[(self.df_wrist["Timestamp"] > start) & (self.df_wrist["Timestamp"] < stop)]

            if downsample_factor != 1:
                df_wrist = df_wrist.iloc[::downsample_factor, :]

        # Window length in minutes
        try:
            window_len = (stop - start).seconds / 60
        except TypeError:
            window_len = (datetime.strptime(stop, "%Y-%m-%d %H:%M:%S") -
                          datetime.strptime(start, "%Y-%m-%d %H:%M:%S")).seconds / 60

        print("Plotting {} minute section from {} to {}.".format(window_len, start, stop))

        # Downsampling information ------------------------------------------------------------------------------------
        if downsample_factor != 1:
            if self.wrist_fname is not None:
                print("\nDownsampling {}Hz data by a factor of {}. "
                      "New data is {}Hz.".format(self.wrist_samplerate, downsample_factor,
                                                 round(self.wrist_samplerate / downsample_factor, 1)))
            if self.wrist_fname is None:
                print("\nDownsampling {}Hz data by a factor of {}. "
                      "New data is {}Hz.".format(self.hip_samplerate, downsample_factor,
                                                 round(self.hip_samplerate / downsample_factor, 1)))

        # Formatting x-axis ticks ------------------------------------------------------------------------------------
        xfmt = mdates.DateFormatter("%a %b %d, %H:%M:%S")

        # Generates ~15 ticks (1/15th of window length apart)
        locator = mdates.MinuteLocator(byminute=np.arange(0, 59, int(np.ceil(window_len / 15))), interval=1)

        # Plots depending on what data is available -------------------------------------------------------------------
        if self.hip_fname is not None and self.wrist_fname is not None:

            def plot_wrist_hip():
                fig, (ax1, ax2) = plt.subplots(2, sharex='col')
                plt.subplots_adjust(bottom=.17)

                ax1.set_title("{}".format(self.wrist_fname))
                ax1.plot(df_wrist["Timestamp"], df_wrist["X"], color='red', label="Wrist_X")
                ax1.plot(df_wrist["Timestamp"], df_wrist["Y"], color='black', label="Wrist_Y")
                ax1.plot(df_wrist["Timestamp"], df_wrist["Z"], color='blue', label="Wrist_Z")
                ax1.legend()
                ax1.set_ylabel("G")

                ax2.set_title("{}".format(self.hip_fname.split("/")[-1]))
                ax2.plot(df_hip["Timestamp"], df_hip["X"], color='red', label="hip_X")
                ax2.plot(df_hip["Timestamp"], df_hip["Y"], color='black', label="hip_Y")
                ax2.plot(df_hip["Timestamp"], df_hip["Z"], color='blue', label="hip_Z")
                ax2.legend()
                ax2.set_ylabel("G")

                # Timestamp axis formatting
                xfmt = mdates.DateFormatter("%a %b %d, %H:%M")

                ax2.xaxis.set_major_formatter(xfmt)
                ax2.xaxis.set_major_locator(locator)
                plt.xticks(rotation=45, fontsize=6)

            plot_wrist_hip()

        if self.hip_fname is None and self.wrist_fname is not None:

            def plot_wrist():
                fig, ax1 = plt.subplots(1)
                plt.subplots_adjust(bottom=.17)

                ax1.set_title("{}".format(self.wrist_fname.split("/")[-1]))
                ax1.plot(df_wrist["Timestamp"], df_wrist["X"], color='red', label="Wrist_X")
                ax1.plot(df_wrist["Timestamp"], df_wrist["Y"], color='black', label="Wrist_Y")
                ax1.plot(df_wrist["Timestamp"], df_wrist["Z"], color='blue', label="Wrist_Z")
                ax1.legend()
                ax1.set_ylabel("G")

                # Timestamp axis formatting
                xfmt = mdates.DateFormatter("%a %b %d, %H:%M")

                ax1.xaxis.set_major_formatter(xfmt)
                ax1.xaxis.set_major_locator(locator)
                plt.xticks(rotation=45, fontsize=6)

            plot_wrist()

        if self.hip_fname is not None and self.wrist_fname is None:

            def plot_hip():
                fig, ax1 = plt.subplots(1)
                plt.subplots_adjust(bottom=.17)

                ax1.set_title("{}".format(self.hip_fname.split("/")[-1]))
                ax1.plot(df_hip["Timestamp"], df_hip["X"], color='red', label="hip_X")
                ax1.plot(df_hip["Timestamp"], df_hip["Y"], color='black', label="hip_Y")
                ax1.plot(df_hip["Timestamp"], df_hip["Z"], color='blue', label="hip_Z")
                ax1.legend()
                ax1.set_ylabel("G")

                # Timestamp axis formatting

                ax1.xaxis.set_major_formatter(xfmt)
                ax1.xaxis.set_major_locator(locator)
                plt.xticks(rotation=45, fontsize=6)

            plot_hip()

    def compare_filter(self, start=None, stop=None, data_type=None, downsample_factor=1):
        """Plots raw and filtered data on separate subplots.

        arguments:
            -start: timestamp for start of region. Format = "YYYY-MM-DD HH:MM:SS"
            -stop: timestamp for end of region. Format = "YYYY-MM-DD HH:MM:SS"
            -downsample_factor: ratio by which data are downsampled. E.g. downsample=3 will downsample from 75 to 25 Hz

        If start and stop are not specified, data will be cropped to one of the following:
            -If no previous graphs have been generated, it will plot the entire data file
            -If a previous crop has occurred, it will 'remember' that region and plot it again.

        To clear the 'memory' of previously-plotted regions, enter "x.start_stamp=None"
        and "x.stop_stop=None" in console
        """

        print("\n-----------------------------------------------------------------------------------------------------")

        # Sets which data to use
        if data_type == "hip" or data_type == "Hip":
            df = self.df_hip
        if data_type == "wrist" or data_type == "Wrist":
            df = self.df_wrist

        # Formats arguments as datetimes -----------------------------------------------------------------------------

        # If arguments are given and no previous region has been specified
        if start is not None and self.start_stamp is None:
            start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        if stop is not None and self.stop_stamp is None:
            stop = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S")

        # If arguments not given and no stamps from previous region
        # Sets start/stop to first/last timestamp for hip or wrist data
        if start is None and self.start_stamp is None:
            try:
                start = self.df_hip["Timestamp"].iloc[0]
            except (AttributeError, ValueError):
                start = self.df_wrist["Timestamp"].iloc[0]

        if stop is None and self.stop_stamp is None:
            try:
                stop = self.df_hip["Timestamp"].iloc[-1]
            except (AttributeError, ValueError):
                stop = self.df_wrist["Timestamp"].iloc[-1]

        # If arguments are not given and there are stamps from previous region
        if start is None and self.start_stamp is not None:
            print("Plotting previously-plotted region.")
            start = self.start_stamp
        if stop is None and self.stop_stamp is not None:
            stop = self.stop_stamp

        # Sets 'memory' values to current start/stop values
        self.start_stamp = start
        self.stop_stamp = stop

        # Crops dataframes to selected region
        df = df.loc[(df["Timestamp"] > start) & (df["Timestamp"] < stop)]

        # Downsamples data
        if downsample_factor != 1:
            df = df.iloc[::downsample_factor, :]

            if data_type == "wrist" or data_type == "Wrist":
                print("\nDownsampling {}Hz data by a factor of {}. "
                      "New data is {}Hz.".format(self.wrist_samplerate, downsample_factor,
                                                 round(self.wrist_samplerate / downsample_factor, 1)))

            if data_type == "hip" or data_type == "Hip":
                print("\nDownsampling {}Hz data by a factor of {}. "
                      "New data is {}Hz.".format(self.hip_samplerate, downsample_factor,
                                                 round(self.hip_samplerate / downsample_factor, 1)))

        # Window length in minutes
        try:
            window_len = (stop - start).seconds / 60
        except TypeError:
            window_len = (datetime.strptime(stop, "%Y-%m-%d %H:%M:%S") -
                          datetime.strptime(start, "%Y-%m-%d %H:%M:%S")).seconds / 60

        print("Plotting {} minute section from {} to {}.".format(window_len, start, stop))

        # Formatting x-axis ticks ------------------------------------------------------------------------------------
        xfmt = mdates.DateFormatter("%a %b %d, %H:%M:%S")

        # Generates ~15 ticks (1/15th of window length apart)
        locator = mdates.MinuteLocator(byminute=np.arange(0, 59, int(np.ceil(window_len / 15))), interval=1)

        # PLOTTING ----------------------------------------------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
        plt.subplots_adjust(bottom=.17)

        ax1.set_title("Raw {} data".format(data_type))
        ax1.plot(df["Timestamp"], df["X"], label="X")
        ax1.plot(df["Timestamp"], df["Y"], label="Y")
        ax1.plot(df["Timestamp"], df["Z"], label="Z")
        ax1.set_ylabel("G's")
        ax1.legend()

        ax2.set_title("Filtered {} data".format(data_type))
        ax2.plot(df["Timestamp"], df["X_filt"], label="X_filt")
        ax2.plot(df["Timestamp"], df["Y_filt"], label="Y_filt")
        ax2.plot(df["Timestamp"], df["Z_filt"], label="Z_filt")
        ax2.set_ylabel("G's")
        ax2.legend()

        ax2.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)


# Creates data objects
x = GENEActiv(hip_filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/Test_Ankle.EDF")

# ADDITIONAL FUNCTIONS TO RUN -----------------------------------------------------------------------------------------

# Filtering
# x.filter_signal(data_type='hip', type="bandpass", low_f=1, high_f=10, sample_f=75, filter_order=5)

# Plots section of data between start and stop arguments. Formatted as YYYY-MM-DD HH:MM:SS
# x.plot_data(start="2019-10-03 10:34:00", stop="2019-10-03 11:15:00", downsample_factor=1) # Section of data
# x.plot_data(start=None, stop=None, downsample_factor=1)  # Plots whole file OR plots region previously specified

# Plots raw and filtered data on separate subplots
# x.compare_filter(start="2019-10-03 10:34:00", stop="2019-10-03 10:59:00", data_type="hip")
# x.compare_filter(data_type="hip", downsample_factor=1)

# Clearing data cropping 'memory'
# x.start_stamp, x.stop_stamp = None, None
