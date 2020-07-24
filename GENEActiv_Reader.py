import pyedflib  # wavelet toolbox for reading / writing EDF/BDF files
import numpy as np  ## package for scientific computing
import pandas as pd  # package for data analysis and manipulation tools
from datetime import datetime  # module supplying classes for manipulating dates and times
from datetime import timedelta
import matplotlib.pyplot as plt  # library for creating static, animated, and interactive visualizations
import matplotlib.dates as mdates
import os  # module allowing code to use operating system dependent functionality
from scipy.signal import butter, filtfilt  # signal processing toolbox
import peakutils  # package for utilities related to the detection of peaks on 2D data


class GENEActiv:

    # ==================================================== BLOCK 2A ===================================================
    # This block defines our method(s) for loading our accelerometer files

    def __init__(self, wrist_filepath=None, hip_filepath=None, leftankle_filepath=None, rightankle_filepath=None,
                 ecg_filepath=None, fig_height=7, fig_width=12):
        """Class that reads in GENEActiv data (EDF). Data is read in and no further methods are called.
            :arguments:
            -wrist_filepath, hip_filepath, leftankle_filepath, rightright_filepath: full pathway to all .edf files
             to read in. Default value is None; fill will not be read in if no argument given
             -fig_heigth, fig_width: figure height and width in inches (I think). Must be whole number.
            """

        self.wrist_fname = wrist_filepath
        self.hip_fname = hip_filepath
        self.lankle_fname = leftankle_filepath
        self.rankle_fname = rightankle_filepath
        self.ecg_fname = ecg_filepath

        self.fig_height = fig_height
        self.fig_width = fig_width

        self.accel_filter_low_f = None
        self.accel_filter_low_h = None
        self.ecg_filter_freq_l = None
        self.ecg_filter_freq_h = None

        self.df_wrist, self.wrist_samplerate = self.load_correct_file(filepath=self.wrist_fname, f_type="Wrist")
        self.df_hip, self.hip_samplerate = self.load_correct_file(filepath=self.hip_fname, f_type="Hip")
        self.df_lankle, self.lankle_samplerate = self.load_correct_file(filepath=self.lankle_fname,
                                                                        f_type="Left ankle")
        self.df_rankle, self.rankle_samplerate = self.load_correct_file(filepath=self.rankle_fname,
                                                                        f_type="Right ankle")
        self.df_ecg, self.ecg_samplerate = self.import_edf(filepath=self.ecg_fname, f_type="ECG")

        self.df_epoched = None

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
    def import_edf(filepath=None, f_type="Accelerometer"):
        """Method that imports EDF from specified filepath and returns a df of timestamps and x, y, z channels.
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

        if filepath is not None and f_type == "ECG":
            print("------------------------------------- {} data -------------------------------------".format(f_type))
            print("\nImporting {}...".format(filepath))

            # READS IN EG DATA ========================================================================================
            file = pyedflib.EdfReader(filepath)
            sample_rate = file.getSampleFrequencies()[0]
            starttime = file.getStartdatetime()

            ecg = file.readSignal(chn=0)

            file.close()

            # TIMESTAMP GENERATION ===================================================================================
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

    # ==================================================== BLOCK 2B ===================================================
    # This block defines our method(s) for epoching our data to calculate activity counts

    def epoch_data(self, epoch_length=15):
        """Creates df of epoched data for all available devices. Able to set epoch_length in seconds."""

        print("\nCalculating activity counts for epochs of length {} seconds...".format(epoch_length))

        svm_lists = []
        devices = ["Timestamp"]

        for data_type in ["wrist", "hip", "lankle", "rankle"]:

            # DATA SET UP
            if data_type == "wrist" and self.wrist_fname is not None:
                data = self.df_wrist["Mag"]
                fs = self.wrist_samplerate
                devices.append(data_type.capitalize())
                timestamps = self.df_wrist["Timestamp"]

            if data_type == "hip" and self.hip_fname is not None:
                data = self.df_hip["Mag"]
                fs = self.hip_samplerate
                devices.append(data_type.capitalize())
                timestamps = self.df_hip["Timestamp"]

            if data_type == "lankle" and self.lankle_fname is not None:
                data = self.df_lankle["Mag"]
                fs = self.lankle_samplerate
                devices.append(data_type.capitalize())
                timestamps = self.df_lankle["Timestamp"]

            if data_type == "rankle" and self.rankle_fname is not None:
                data = self.df_rankle["Mag"]
                fs = self.rankle_samplerate
                devices.append(data_type.capitalize())
                timestamps = self.df_rankle["Timestamp"]

            svm = []

            for i in np.arange(0, data.shape[0], fs * epoch_length):
                svm.append(sum(data.iloc[int(i):int(i + fs)]))

            svm_lists.append(svm)

        df = pd.DataFrame(svm_lists).transpose()

        epoch_stamps = [timestamps.iloc[0] + timedelta(seconds=15) * i for i in range(0, df.shape[0])]

        df.insert(loc=0, column="Timestamp", value=epoch_stamps)
        df.columns = devices

        print("Epoching complete.")

        print("\nEpoched data device Pearson correlation matrix:")

        print(df.corr())

        return df

    # ==================================================== BLOCK 2C ===================================================
    # This block defines our method(s) for filtering data

    def filter_signal(self, data_type="accelerometer", type="bandpass", low_f=1, high_f=10, filter_order=1):
        """Filtering details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
        Arguments:
            -type: filter type - "bandpass", "lowpass", or "highpass"
            -low_f: low-end cutoff frequency, required for lowpass and bandpass filters
            -high_f: high-end cutoff frequency, required for highpass and bandpass filters
            -filter_order: integet for filter order
        Adds columns to dataframe corresponding to each device. Filters all devices that are available.
        """

        if data_type == "accelerometer":

            self.accel_filter_low_f = low_f
            self.accel_filter_low_h = high_f

            for data_type in ["wrist", "hip", "lankle", "rankle"]:

                # DATA SET UP
                if data_type == "wrist" and self.wrist_fname is not None:
                    data = np.array([self.df_wrist["X"], self.df_wrist["Y"], self.df_wrist["Z"]])
                    original_df = self.df_wrist
                    fs = self.wrist_samplerate * .5

                if data_type == "hip" and self.hip_fname is not None:
                    data = np.array([self.df_hip["X"], self.df_hip["Y"], self.df_hip["Z"]])
                    original_df = self.df_hip
                    fs = self.hip_samplerate * .5

                if data_type == "lankle" and self.lankle_fname is not None:
                    data = np.array([self.df_lankle["X"], self.df_lankle["Y"], self.df_lankle["Z"]])
                    original_df = self.df_lankle
                    fs = self.lankle_samplerate * .5

                if data_type == "rankle" and self.rankle_fname is not None:
                    data = np.array([self.df_rankle["X"], self.df_rankle["Y"], self.df_rankle["Z"]])
                    original_df = self.df_rankle
                    fs = self.rankle_samplerate * .5

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

        if data_type == "ECG":
            self.ecg_filter_freq_l = low_f
            self.ecg_filter_freq_h = high_f

            data = np.array(self.df_ecg["Raw"])
            original_df = self.df_ecg
            fs = self.ecg_samplerate * .5

            print("\nFiltering ECG data with {}-{}Hz, order {} bandpass filter.".format(data_type, low_f,
                                                                                        high_f, filter_order))

            low = low_f / fs
            high = high_f / fs
            b, a = butter(N=filter_order, Wn=[low, high], btype="bandpass")
            filtered_data = filtfilt(b, a, x=data)

            original_df["Filtered"] = filtered_data

        print("\nFiltering complete.")

    # ==================================================== BLOCK 2D ===================================================
    # This block defines our method(s) for plotting data

    def plot_data(self, start=None, stop=None, downsample_factor=1):
        """Plots hip and wrist data whichever/both is available.
            arguments:
                -start: timestamp for start of region. Format = "YYYY-MM-DD HH:MM:SS" OR integer for
                        minutes into collection
                -stop: timestamp for end of region. Format = "YYYY-MM-DD HH:MM:SS" OR integer for
                        minutes into collection
                -downsample: ratio by which data are downsampled. E.g. downsample=3 will downsample from 75 to 25 Hz
            If start and stop are not specified, data will be cropped to one of the following:
                -If no previous graphs have been generated, it will plot the entire data file
                -If a previous crop has occurred, it will 'remember' that region and plot it again.
            To clear the 'memory' of previously-plotted regions, enter "x.start_stamp=None"
            and "x.stop_stop=None" in console
        """

        print("\n-----------------------------------------------------------------------------------------------------")

        # Gets appropriate timestamps
        start_stamp, stop_stamp, data_type = self.get_timestamps(start, stop)

        self.start_stamp = start_stamp
        self.stop_stamp = stop_stamp

        # Crops dataframes to selected region -------------------------------------------------------------------------
        if self.hip_fname is not None:
            # Sets stop to end of collection if stop timestamp exceeds timestamp range
            try:
                if stop_stamp > self.df_hip.iloc[-1]["Timestamp"]:
                    stop_stamp = self.df_hip.iloc[-1]["Timestamp"]
            except TypeError:
                if datetime.strptime(stop_stamp, "%Y-%m-%d %H:%M:%S") > self.df_hip.iloc[-1]["Timestamp"]:
                    stop_stamp = self.df_hip.iloc[-1]["Timestamp"]

            df_hip = self.df_hip.loc[(self.df_hip["Timestamp"] > start_stamp) & (self.df_hip["Timestamp"] < stop_stamp)]

            if data_type == "absolute":
                df_hip["Timestamp"] = np.arange(0, (stop_stamp - start_stamp).seconds,
                                                1 / self.hip_samplerate)[0:df_hip.shape[0]]

            if downsample_factor != 1:
                df_hip = df_hip.iloc[::downsample_factor, :]

        if self.wrist_fname is not None:
            # Sets stop to end of collection if stop timestamp exceeds timestamp range
            try:
                if stop_stamp > self.df_wrist.iloc[-1]["Timestamp"]:
                    stop_stamp = self.df_wrist.iloc[-1]["Timestamp"]
            except TypeError:
                if datetime.strptime(stop_stamp, "%Y-%m-%d %H:%M:%S") > self.df_wrist.iloc[-1]["Timestamp"]:
                    stop_stamp = self.df_wrist.iloc[-1]["Timestamp"]

            df_wrist = self.df_wrist.loc[(self.df_wrist["Timestamp"] > start_stamp) &
                                         (self.df_wrist["Timestamp"] < stop_stamp)]

            if data_type == "absolute":
                df_wrist["Timestamp"] = np.arange(0, (stop_stamp - start_stamp).seconds,
                                                  1 / self.wrist_samplerate)[0:df_wrist.shape[0]]

            if downsample_factor != 1:
                df_wrist = df_wrist.iloc[::downsample_factor, :]

        # Window length in minutes
        window_len = (stop_stamp - start_stamp).seconds / 60

        print("Plotting {} minute section from {} to {}.".format(round(window_len, 1), start_stamp, stop_stamp))

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
                fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(self.fig_width, self.fig_height))
                plt.subplots_adjust(bottom=.17, hspace=.33)

                ax1.set_title("{}".format(self.hip_fname.split("/")[-1]))
                ax1.plot(df_wrist["Timestamp"], df_wrist["X"], color='red', label="Wrist_X")
                ax1.plot(df_wrist["Timestamp"], df_wrist["Y"], color='black', label="Wrist_Y")
                ax1.plot(df_wrist["Timestamp"], df_wrist["Z"], color='dodgerblue', label="Wrist_Z")
                ax1.legend(loc='lower left')
                ax1.set_ylabel("G")

                ax2.set_title("{}".format(self.hip_fname.split("/")[-1]))
                ax2.plot(df_hip["Timestamp"], df_hip["X"], color='red', label="Hip_X")
                ax2.plot(df_hip["Timestamp"], df_hip["Y"], color='black', label="Hip_Y")
                ax2.plot(df_hip["Timestamp"], df_hip["Z"], color='dodgerblue', label="Hip_Z")
                ax2.legend(loc='lower left')
                ax2.set_ylabel("G")

                # Timestamp axis formatting
                if data_type == "timestamp":
                    xfmt = mdates.DateFormatter("%a %b %d, %H:%M")

                    ax2.xaxis.set_major_formatter(xfmt)
                    ax2.xaxis.set_major_locator(locator)
                    plt.xticks(rotation=45, fontsize=8)

                if data_type == "absolute":
                    ax2.set_xlabel("Seconds into collection")

            plot_wrist_hip()

        if self.hip_fname is None and self.wrist_fname is not None:

            def plot_wrist():
                fig, ax1 = plt.subplots(1, figsize=(self.fig_width, self.fig_height))
                plt.subplots_adjust(bottom=.17)

                ax1.set_title("{}".format(self.wrist_fname.split("/")[-1]))
                ax1.plot(df_wrist["Timestamp"], df_wrist["X"], color='red', label="Wrist_X")
                ax1.plot(df_wrist["Timestamp"], df_wrist["Y"], color='black', label="Wrist_Y")
                ax1.plot(df_wrist["Timestamp"], df_wrist["Z"], color='dodgerblue', label="Wrist_Z")
                ax1.legend(loc='lower left')
                ax1.set_ylabel("G")

                # Timestamp axis formatting
                xfmt = mdates.DateFormatter("%a %b %d, %H:%M")

                if data_type == "timestamp":
                    ax1.xaxis.set_major_formatter(xfmt)
                    ax1.xaxis.set_major_locator(locator)
                    plt.xticks(rotation=45, fontsize=8)

                if data_type == "absolute":
                    ax1.set_xlabel("Seconds into collection")

            plot_wrist()

        if self.hip_fname is not None and self.wrist_fname is None:

            def plot_hip():
                fig, ax1 = plt.subplots(1, figsize=(self.fig_width, self.fig_height))
                plt.subplots_adjust(bottom=.17)

                ax1.set_title("{}".format(self.hip_fname.split("/")[-1]))
                ax1.plot(df_hip["Timestamp"], df_hip["X"], color='red', label="Hip_X")
                ax1.plot(df_hip["Timestamp"], df_hip["Y"], color='black', label="Hip_Y")
                ax1.plot(df_hip["Timestamp"], df_hip["Z"], color='dodgerblue', label="Hip_Z")
                ax1.legend(loc='lower left')
                ax1.set_ylabel("G")

                # Timestamp axis formatting
                if data_type == "timestamp":
                    ax1.xaxis.set_major_formatter(xfmt)
                    ax1.xaxis.set_major_locator(locator)
                    plt.xticks(rotation=45, fontsize=8)

                if data_type == "absolute":
                    ax1.set_xlabel("Seconds into collection")

            plot_hip()

        plt.savefig("HipWrist_{} to {}.png".format(datetime.strftime(start_stamp, "%Y-%m-%d %H_%M_%S"),
                                                   datetime.strftime(stop_stamp, "%Y-%m-%d %H_%M_%S")))
        print("Plot saved as png (HipWrist_{} to {}.png)".format(datetime.strftime(start_stamp, "%Y-%m-%d %H_%M_%S"),
                                                                 datetime.strftime(stop_stamp, "%Y-%m-%d %H_%M_%S")))

    # ==================================================== BLOCK 2E ===================================================
    # This block defines our method(s) for comparing filtered to unfiltered data

    def compare_filter(self, device_type=None, start=None, stop=None, downsample_factor=1):
        """Plots raw and filtered data on separate subplots.
        arguments:
            -device_type: "hip", "wrist", "lankle", "rankle" or "ECG" --> which device to plot
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
        if device_type == "hip" or device_type == "Hip":
            df = self.df_hip
            fs = self.hip_samplerate
        if device_type == "wrist" or device_type == "Wrist":
            df = self.df_wrist
            fs = self.wrist_samplerate
        if device_type == "lankle" or device_type == "Lankle" or device_type == "LAnkle":
            df = self.df_lankle
            fs = self.lankle_samplerate
        if device_type == "rankle" or device_type == "Rankle" or device_type == "RAnkle":
            df = self.df_rankle
            fs = self.rankle_samplerate
        if device_type == "ECG" or device_type == "ecg":
            df = self.df_ecg
            fs = self.ecg_samplerate

        # Gets appropriate timestamps
        start, stop, data_type = self.get_timestamps(start, stop)

        # Sets 'memory' values to current start/stop values
        self.start_stamp = start
        self.stop_stamp = stop

        # Crops dataframes to selected region
        df = df.loc[(df["Timestamp"] > start) & (df["Timestamp"] < stop)]

        # Downsamples data
        if downsample_factor != 1:
            df = df.iloc[::downsample_factor, :]

            if device_type == "wrist" or device_type == "Wrist":
                print("\nDownsampling {}Hz data by a factor of {}. "
                      "New data is {}Hz.".format(self.wrist_samplerate, downsample_factor,
                                                 round(self.wrist_samplerate / downsample_factor, 1)))

            if device_type == "hip" or device_type == "Hip":
                print("\nDownsampling {}Hz data by a factor of {}. "
                      "New data is {}Hz.".format(self.hip_samplerate, downsample_factor,
                                                 round(self.hip_samplerate / downsample_factor, 1)))

            if device_type == "lankle" or device_type == "Lankle" or device_type == "LAnkle":
                print("\nDownsampling {}Hz data by a factor of {}. "
                      "New data is {}Hz.".format(self.lankle_samplerate, downsample_factor,
                                                 round(self.lankle_samplerate / downsample_factor, 1)))

            if device_type == "rankle" or device_type == "Rankle" or device_type == "RAnkle":
                print("\nDownsampling {}Hz data by a factor of {}. "
                      "New data is {}Hz.".format(self.rankle_samplerate, downsample_factor,
                                                 round(self.rankle_samplerate / downsample_factor, 1)))
            if device_type == "ECG" or device_type == "ecg":
                print("\nDownsampling {}Hz data by a factor of {}. "
                      "New data is {}Hz.".format(self.ecg_samplerate, downsample_factor,
                                                 round(self.ecg_samplerate / downsample_factor, 1)))

        # Window length in minutes
        try:
            window_len = (stop - start).seconds / 60
        except TypeError:
            window_len = (datetime.strptime(stop, "%Y-%m-%d %H:%M:%S") -
                          datetime.strptime(start, "%Y-%m-%d %H:%M:%S")).seconds / 60

        if data_type == "absolute":
            df["Timestamp"] = np.arange(0, (stop - start).seconds, 1 / fs)[0:df.shape[0]]

        print("Plotting {} data: {} minute section from {} to {}.".format(device_type, window_len, start, stop))

        # Formatting x-axis ticks ------------------------------------------------------------------------------------
        xfmt = mdates.DateFormatter("%a %b %d, %H:%M:%S")

        # Generates ~15 ticks (1/15th of window length apart)
        locator = mdates.MinuteLocator(byminute=np.arange(0, 59, int(np.ceil(window_len / 15))), interval=1)

        # ACCELEROMETER PLOTTING -------------------------------------------------------------------------------------
        if device_type != "ECG" and device_type != "ecg":
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(self.fig_width, self.fig_height))
            plt.subplots_adjust(bottom=.17, hspace=.33)

            ax1.set_title("Raw {} data".format(data_type))
            ax1.plot(df["Timestamp"], df["X"], label="X", color='red')
            ax1.plot(df["Timestamp"], df["Y"], label="Y", color='black')
            ax1.plot(df["Timestamp"], df["Z"], label="Z", color='dodgerblue')
            ax1.set_ylabel("G's")
            ax1.legend(loc='lower left')

            ax2.set_title("Filtered {} data".format(data_type))
            ax2.plot(df["Timestamp"], df["X_filt"], label="X_filt", color='red')
            ax2.plot(df["Timestamp"], df["Y_filt"], label="Y_filt", color='black')
            ax2.plot(df["Timestamp"], df["Z_filt"], label="Z_filt", color='dodgerblue')
            ax2.set_ylabel("G's")
            ax2.legend(loc='lower left')

            if data_type == "timestamp":
                ax2.xaxis.set_major_formatter(xfmt)
                ax2.xaxis.set_major_locator(locator)
            if data_type == "absolute":
                ax2.set_xlabel("Seconds into collection")

        # ECG PLOTTING -----------------------------------------------------------------------------------------------
        if device_type == "ECG" or device_type == "ecg":
            fig, ax1 = plt.subplots(1, figsize=(self.fig_width, self.fig_height))

            ax1.set_title("Raw and Filtered ({}-{} Hz bandpass) ECG data".format(self.ecg_filter_freq_l,
                                                                                 self.ecg_filter_freq_h))
            ax1.plot(df["Timestamp"], df["Raw"], color='red', label="Raw")
            ax1.plot(df["Timestamp"], df["Filtered"], color='black', label="Filtered")
            ax1.legend(loc='best')
            ax1.set_ylabel("Voltage")

        if data_type == "timestamp":
            ax1.xaxis.set_major_formatter(xfmt)
            ax1.xaxis.set_major_locator(locator)
        if data_type == "absolute":
            ax1.set_xlabel("Seconds into collection")

        plt.xticks(rotation=45, fontsize=8)

        plt.savefig("{}_RawAndFiltered_{} to {}.png".format(device_type.capitalize(),
                                                            datetime.strftime(start, "%Y-%m-%d %H-%M-%S"),
                                                            datetime.strftime(stop, "%Y-%m-%d %H-%M-%S")))
        print("Plot saved as png ({}_RawAndFiltered_{} to {}.png)".format(device_type.capitalize(),
                                                                          datetime.strftime(start, "%Y-%m-%d %H-%M-%S"),
                                                                          datetime.strftime(stop, "%Y-%m-%d %H-%M-%S")))

    # ==================================================== BLOCK 2F ===================================================
    # This block defines our method(s) for identifying timestamps (previously used ones to replot same data regions)

    def get_timestamps(self, start=None, stop=None):

        # If start/stop given as integer, sets start/stop stamps to minutes into collection ---------------------------
        if type(start) is int and type(stop) is int:

            data_type = "absolute"

            try:
                start_stamp = self.df_hip["Timestamp"].iloc[0] + timedelta(minutes=start)
            except (AttributeError, ValueError, TypeError):
                start_stamp = self.df_wrist["Timestamp"].iloc[0] + timedelta(minutes=start)

            try:
                stop_stamp = self.df_hip["Timestamp"].iloc[0] + timedelta(minutes=stop)
            except (AttributeError, ValueError, TypeError):
                stop_stamp = self.df_wrist["Timestamp"].iloc[0] + timedelta(minutes=stop)

        # Formats arguments as datetimes -----------------------------------------------------------------------------

        # If arguments are given and no previous region has been specified
        else:

            data_type = "timestamp"

            if start is not None and self.start_stamp is None:
                start_stamp = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
            if stop is not None and self.stop_stamp is None:
                stop_stamp = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S")

            # If arguments not given and no stamps from previous region
            # Sets start/stop to first/last timestamp for hip or wrist data
            if start is None and self.start_stamp is None:
                try:
                    start_stamp = self.df_hip["Timestamp"].iloc[0]
                except (AttributeError, ValueError, TypeError):
                    start_stamp = self.df_wrist["Timestamp"].iloc[0]
            if stop is None and self.stop_stamp is None:
                try:
                    stop_stamp = self.df_hip["Timestamp"].iloc[-1]
                except (AttributeError, ValueError, TypeError):
                    stop_stamp = self.df_wrist["Timestamp"].iloc[-1]

            # If arguments are not given and there are stamps from previous region
            if start is None and self.start_stamp is not None:
                print("Plotting previously-plotted region.")
                start_stamp = self.start_stamp
            if stop is None and self.stop_stamp is not None:
                stop_stamp = self.stop_stamp

            # If arguments given --> overrides stamps from previous region
            if start is not None and self.start_stamp is not None:
                start_stamp = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
            if stop is not None and self.stop_stamp is not None:
                stop_stamp = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S")

        return start_stamp, stop_stamp, data_type

    # ==================================================== BLOCK 2G ===================================================
    # This block defines our method(s) for basic peak detection using thresholding

    def plot_peaks(self, signal="X", thresh_type="normalized",
                   peak_thresh=0.5, min_peak_dist=500, downsample_factor=1,
                   start=None, stop=None):
        """"Function that runs peakutils to detect peaks in accelerometer data and plots results.
            :argument
            -signal: what data to run the peak detection on. Must be a column name contained within df_wrist or df_hip
                -Options: "X", "Y", "Z", "Mag", "X_filt", "Y_filt", "Z_filt"
            -thresh_type: either "normalized" or "absolute".
            -peak_threshold: value to set threshold based on thresh_type. If thresh_type="normalized", peak_threshold
                             is a value from 0 to 1 that represents the threshold as a percent of the signal amplitude.
                             If thresh_type="absolute", the threshold can be any value that corresponds to G's
                -Threshold is calculated as (max - min) * threshold + min
            -min_peak_dist: number of milliseconds required between consecutive peaks
            -start: timestamp for start of region. Format = "YYYY-MM-DD HH:MM:SS"
            -stop: timestamp for end of region. Format = "YYYY-MM-DD HH:MM:SS"
            -downsample_factor: ratio by which data are downsampled. E.g. downsample=3 will downsample from 75 to 25 Hz
        If start and stop are not specified, data will be cropped to one of the following:
        -If no previous graphs have been generated, it will plot the entire data file
        -If a previous crop has occurred, it will 'remember' that region and plot it again.
        To clear the 'memory' of previously-plotted regions, enter "x.start_stamp=None"
        and "x.stop_stop=None" in console
        """

        # Gets appropriate timestamps
        start, stop, data_type = self.get_timestamps(start, stop)

        # Window length in minutes
        try:
            window_len = (stop - start).seconds / 60
        except TypeError:
            window_len = (datetime.strptime(stop, "%Y-%m-%d %H:%M:%S") -
                          datetime.strptime(start, "%Y-%m-%d %H:%M:%S")).seconds / 60

        print("\nPlotting {} minute section from {} to {}.".format(window_len, start, stop))

        # Sets 'memory' values to current start/stop values
        self.start_stamp = start
        self.stop_stamp = stop

        # PEAK DETECTION ---------------------------------------------------------------------------------------------

        if self.hip_fname is not None:
            hip_data = self.df_hip.loc[(self.df_hip["Timestamp"] > start) & (self.df_hip["Timestamp"] < stop)]
            hip_data = hip_data.iloc[::downsample_factor, :]

            hip_fs = self.hip_samplerate

            hip_peaks = peakutils.indexes(y=np.array(hip_data[signal]),
                                          thres_abs=True if thresh_type == "absolute" else False,
                                          thres=peak_thresh,
                                          min_dist=int(min_peak_dist / downsample_factor / (1000 / hip_fs)))

            print("-Hip accelerometer: found {} steps.".format(len(hip_peaks)))

            if data_type == "absolute":
                hip_data["Timestamp"] = np.arange(0, (stop - start).seconds,
                                                  1 / self.hip_samplerate)[0:hip_data.shape[0]]

        if self.lankle_fname is not None:
            la_data = self.df_lankle.loc[(self.df_lankle["Timestamp"] >= start) & (self.df_lankle["Timestamp"] < stop)]
            la_data = la_data.iloc[::downsample_factor, :]
            la_fs = self.lankle_samplerate

            la_peaks = peakutils.indexes(y=np.array(la_data[signal]),
                                         thres_abs=True if thresh_type == "absolute" else False,
                                         thres=peak_thresh,
                                         min_dist=int(min_peak_dist / downsample_factor / (1000 / la_fs)))

            print("-Left ankle accelerometer: found {} steps.".format(len(la_peaks)))

            if data_type == "absolute":
                la_data["Timestamp"] = np.arange(0, (stop - start).seconds, 1 / self.hip_samplerate)[0:la_data.shape[0]]

        if self.rankle_fname is not None:
            ra_data = self.df_rankle.loc[(self.df_rankle["Timestamp"] >= start) & (self.df_rankle["Timestamp"] < stop)]
            ra_data = ra_data.iloc[::downsample_factor, :]
            ra_fs = self.rankle_samplerate

            ra_peaks = peakutils.indexes(y=np.array(ra_data[signal]),
                                         thres_abs=True if thresh_type == "absolute" else False,
                                         thres=peak_thresh,
                                         min_dist=int(min_peak_dist / downsample_factor / (1000 / ra_fs)))

            print("-Right ankle accelerometer: found {} steps.".format(len(ra_peaks)))

            if data_type == "absolute":
                ra_data["Timestamp"] = np.arange(0, (stop - start).seconds, 1 / self.hip_samplerate)[0:ra_data.shape[0]]

        # PLOTTING ---------------------------------------------------------------------------------------------------

        # Determines number of subplots and formatting
        if self.hip_fname is not None and (self.lankle_fname is not None or self.rankle_fname is not None):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(self.fig_width, self.fig_height))

            ax2.set_ylabel("G")

            if thresh_type == "absolute":
                ax1.axhline(y=peak_thresh, color='green', linestyle='dashed', label="Thresh={} G".format(peak_thresh))
                ax2.axhline(y=peak_thresh, color='green', linestyle='dashed', label="Thresh={} G".format(peak_thresh))

        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))

            if thresh_type == "absolute":
                ax1.axhline(y=peak_thresh, color='green', linestyle='dashed', label="Thresh={} G".format(peak_thresh))

            xfmt = mdates.DateFormatter("%a %b %d, %H:%M:%S")
            ax1.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45, fontsize=8)

        plt.subplots_adjust(bottom=.15)

        # HIP DATA -------------------------------------
        ax1.plot(hip_data["Timestamp"], hip_data[signal],
                 color='black', label="Hip_{}".format(signal))

        ax1.plot([hip_data["Timestamp"].iloc[i] for i in hip_peaks],
                 [hip_data[signal].iloc[i] for i in hip_peaks],
                 color='red', marker="o", linestyle="", label="Peaks (n={})".format(len(hip_peaks)))

        ax1.set_ylabel("G")
        ax1.set_title("Signal={}, peak_thresh={}, "
                      "thresh_type={}, min_dist={} ms, downsample={}".format(signal, peak_thresh,
                                                                             thresh_type, min_peak_dist,
                                                                             downsample_factor))

        ax1.legend(loc='lower left')

        if self.lankle_fname is None and self.rankle_fname is None:
            xfmt = mdates.DateFormatter("%a %b %d, %H:%M:%S")
            ax1.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45, fontsize=8)

        # ANKLE DATA ------------------------------------
        if self.lankle_fname is not None and self.rankle_fname is not None:

            # Left ankle
            ax2.plot(la_data["Timestamp"], la_data[signal],
                     color='dodgerblue', label="LA_{}".format(signal))

            ax2.plot([la_data["Timestamp"].iloc[i] for i in la_peaks],
                     [la_data[signal].iloc[i] for i in la_peaks],
                     color='red', marker="o", linestyle="", label="LA peaks (n={})".format(len(la_peaks)))

            # Right ankle
            ax2.plot(ra_data["Timestamp"], ra_data[signal],
                     color='grey', label="RA_{}".format(signal))

            ax2.plot([ra_data["Timestamp"].iloc[i] for i in ra_peaks],
                     [ra_data[signal].iloc[i] for i in ra_peaks],
                     color='cyan', marker="X", linestyle="", label="RA peaks (n={})".format(len(ra_peaks)))

            ax2.set_xlim(la_data.iloc[0]["Timestamp"], la_data.iloc[-1]["Timestamp"])

            ax2.legend(loc='lower left')

            if data_type == "timestamp":
                xfmt = mdates.DateFormatter("%a %b %d, %H:%M:%S")
                ax2.xaxis.set_major_formatter(xfmt)
                plt.xticks(rotation=45, fontsize=8)

            if data_type == "absolute":
                ax2.set_xlabel("Seconds into collection")

        plt.savefig("StepPeakDetection_{} to {}.png".format(datetime.strftime(start, "%Y-%m-%d %H-%M-%S"),
                                                            datetime.strftime(stop, "%Y-%m-%d %H-%M-%S")))
        print("\nPlot saved as png "
              "(StepPeakDetection_{} to {}.png)".format(datetime.strftime(start, "%Y-%m-%d %H-%M-%S"),
                                                        datetime.strftime(stop, "%Y-%m-%d %H-%M-%S")))

    # ==================================================== BLOCK 2H ===================================================
    # This block defines our method(s) for plotting epoched data (activity counts)

    def plot_epoched(self, start=None, stop=None):
        """Plots epoched data for all available devices.
            :arguments
            -start: timestamp for start of region. Format = "YYYY-MM-DD HH:MM:SS"
            -stop: timestamp for end of region. Format = "YYYY-MM-DD HH:MM:SS"
        If start and stop are not specified, data will be cropped to one of the following:
            -If no previous graphs have been generated, it will plot the entire data file
            -If a previous crop has occurred, it will 'remember' that region and plot it again.
        To clear the 'memory' of previously-plotted regions, enter "x.start_stamp=None"
        and "x.stop_stop=None" in console
        """

        print("\n-----------------------------------------------------------------------------------------------------")

        color_dict = {"Hip": "black", "Wrist": "red", "Lankle": "dodgerblue", "Rankle": "grey"}

        # Gets appropriate timestamps
        start, stop, data_type = self.get_timestamps(start, stop)

        # Sets 'memory' values to current start/stop values
        self.start_stamp = start
        self.stop_stamp = stop

        # Crops dataframes to selected region
        df = self.df_epoched.loc[(self.df_epoched["Timestamp"] > start) & (self.df_epoched["Timestamp"] < stop)]

        # Window length in minutes
        try:
            window_len = (stop - start).seconds / 60
        except TypeError:
            window_len = (datetime.strptime(stop, "%Y-%m-%d %H:%M:%S") -
                          datetime.strptime(start, "%Y-%m-%d %H:%M:%S")).seconds / 60

        # Calculates epoch_length
        epoch_length = (df.iloc[1]["Timestamp"] - df.iloc[0]["Timestamp"]).seconds

        print("Plotting {} minute section of epoched data from {} to {}.".format(window_len, start, stop))

        fig, ax1 = plt.subplots(1, figsize=(self.fig_width, self.fig_height))
        plt.title("Epoched Accelerometer Data")

        for col_name in self.df_epoched.columns[1:]:
            plt.plot(self.df_epoched["Timestamp"], self.df_epoched[col_name],
                     color=color_dict[col_name], label=col_name, marker="x")

        plt.ylabel("Activity counts per {} seconds".format(epoch_length))
        plt.legend()

        # Formatting x-axis ticks ------------------------------------------------------------------------------------
        xfmt = mdates.DateFormatter("%a %b %d, %H:%M:%S")

        # Generates ~15 ticks (1/15th of window length apart)
        print(window_len)
        locator = mdates.MinuteLocator(byminute=np.arange(0, 59, int(np.ceil(window_len / 15))), interval=1)

        ax1.xaxis.set_major_formatter(xfmt)
        ax1.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=8)

        plt.savefig("EpochedData_{} to {}.png".format(datetime.strftime(start, "%Y-%m-%d %H-%M-%S"),
                                                      datetime.strftime(stop, "%Y-%m-%d %H-%M-%S")))

    # ==================================================== BLOCK 2I ===================================================
    # This block defines our method(s) for creating CSV files of our activity counts ###

    def write_epoched_csv(self):

        print("\nWriting epoched data to csv file 'Epoched_ActivityCounts.csv'")
        self.df_epoched.to_csv("Epoched_ActivityCounts.csv", index=False)
        self.df_epoched.to_csv("Epoched_ActivityCounts.csv", index=False)
        print("Complete.")


# ==================================================== BLOCK 3 =====================================================
# This block identifies the files of interest and creates corresponding data objects
"""Change the filepath(s) to the file(s) of interest (e.g. x = GENEActiv(hip_filepath=*"Test_Body.EDF"*)"""
"""
x = GENEActiv(hip_filepath="/Users/kyleweber/Desktop/Data/KW4_GA_LWrist.csv",
              wrist_filepath="/Users/kyleweber/Desktop/Data/KW4_GA_LWrist.csv",
              leftankle_filepath="/Users/kyleweber/Desktop/Data/KW4_GA_LAnkle.csv",
              rightankle_filepath="/Users/kyleweber/Desktop/Data/KW4_GA_RAnkle.csv",
              fig_height=6, fig_width=10)"""
x = GENEActiv(ecg_filepath="/Users/kyleweber/Desktop/Data/KW4_BittiumFaros.EDF",
              wrist_filepath="/Users/kyleweber/Desktop/Data/KW4_GA_LWrist.csv")

# ADDITIONAL FUNCTIONS TO RUN -----------------------------------------------------------------------------------------

# ==================================================== BLOCK 4 =====================================================
# This block applies filtering to signals read from the data files ###
# Run this block if filtering of data is required ###
"""If applicable, change filtering arguments type, low_f, high_f, sample_f, and filter_order"""

# x.filter_signal(data_type="accelerometer", type="bandpass", low_f=1, high_f=10, filter_order=3)
x.filter_signal(data_type="ECG", type="bandpass", low_f=5, high_f=15, filter_order=3)

# ==================================================== BLOCK 5 =====================================================
# This block epochs the signals read from the data files to calculate activity counts ###
# Run this block if activity counts are required ###
# This will print out a Pearson correlation matrix for each body segment ###
"""If applicable, change epoching argument epoch_length"""

# x.df_epoched = x.epoch_data(epoch_length=15)

# ==================================================== BLOCK 6A =====================================================
# This block plots the entire dataset (or if applicable, data within a window based on timestamps) ###
# Run this block if data plotting is required ###
# This will plot data and also generate a .PNG file ###
"""Remove # from line 1 if using specific timestamps (formatted as YYYY-MM-DD HH:MM:SS)"""
"""If applicable, change time argument start, stop; downsample_factor will decrease # of samples"""

# x.plot_data(start="2018-07-03 13:15:00", stop="2018-07-03 13:35:00", downsample_factor=1)  # Section of data
# x.plot_data(start=15, stop=20, downsample_factor=1)  # Plots whole file OR plots region previously specified

# ==================================================== BLOCK 6B =====================================================
# This block clearing data cropping 'memory'by resetting the timestamps ###
"""Remove # from line 1 to apply"""
# x.start_stamp, x.stop_stamp = None, None

# ==================================================== BLOCK 7 =====================================================
# This block creates seprate subplots for filtered vs. unfiltered data based on timestamps specified in Block 6A ###
# Run this block if comparing filtered vs. unfiltered data in plot form is required ###
# This will plot data and also generate a .PNG file ###
"""Remove # from line 1 if applying block of code"""
"""If applicable, change time argument start, stop; downsample_factor will decrease # of samples"""

# x.compare_filter(device_type="lankle", downsample_factor=1)
x.compare_filter(device_type="ECG", downsample_factor=1, start=0, stop=2)

# ==================================================== BLOCK 8 =====================================================
# This block creates plots with peaks identified based on parameters specified in Block 2E ###
# Run this block if plots with peaks detected is required ###
# This will plot data and also generate a .PNG file ###
"""Remove # from line 1 if applying block of code"""
"""If applicable, change peak_thresh argument"""

# x.plot_peaks(signal="X_filt", thresh_type="normalized", peak_thresh=.7, min_peak_dist=400, downsample_factor=1, start=5, stop=15)

# ==================================================== BLOCK 8 =====================================================
# This block creates plots for epoched data ###
# Run this block if activity count plot is required ###
# This will plot data and also generate a .PNG file ###
"""Remove # from line 1 if applying block of code"""
# x.plot_epoched()

# ==================================================== BLOCK 9 =====================================================
# This block creates a delimited spreadsheet file for epoched data ###
# Run this block if a new spreadsheet is required ###
# This will generate a .CSV file ###
"""Remove # from line 1 if applying block of code"""
# x.write_epoched_csv()
