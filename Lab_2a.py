# ==================================================== BLOCK 1 ========================================================

import pyedflib
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from scipy.signal import butter, filtfilt

# ==================================================== BLOCK 2 ========================================================


class Wearables:

    # ==================================================== BLOCK 2A ===================================================

    def __init__(self, hip_filepath=None,
                 fig_height=7, fig_width=12):
        """Class that reads in GENEActiv data.
            Data is read in and no further methods are called.
            :arguments:
            -wrist_filepath, hip_filepath, ankle_filepath, head_filepath:
                    full pathway to all .edf files to read in. Default value is None;
                    fill will not be read in if no argument given
             -fig_heigth, fig_width: figure height and width in inches. Must be whole number.
            """

        # Default values for objects ----------------------------------------------------------------------------------
        self.hip_fname = hip_filepath

        self.fig_height = fig_height
        self.fig_width = fig_width

        self.accel_filter_low_f = None
        self.accel_filter_low_h = None

        # Methods and objects that are run automatically when class instance is created -------------------------------
        self.df_hip, self.hip_samplerate = self.load_correct_file(filepath=self.hip_fname, f_type="Hip")

        # 'Memory' stamps for previously-graphed region
        self.start_stamp = None
        self.stop_stamp = None

        # Details of last filter that was run
        self.filter_details = ""

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
           Works for both GENEACtiv and Bittium Faros EDF files.
           Also returns sampling rate.
           If no file was specified or a None was given, method returns None, None
           """

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

    # ==================================================== BLOCK 2B ===================================================

    def filter_signal(self, device_type="accelerometer", filter_type="bandpass", low_f=1, high_f=10, filter_order=1):
        """Filtering details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
        Arguments:
            -device_type: "accelerometer" or "ECG"
            -filter_type: filter type - "bandpass", "lowpass", or "highpass"
            -low_f: low-end cutoff frequency, required for lowpass and bandpass filters
            -high_f: high-end cutoff frequency, required for highpass and bandpass filters
            -filter_order: integet for filter order
        Adds columns to dataframe corresponding to each device. Filters all devices that are available.
        """

        if device_type == "accelerometer":

            self.accel_filter_low_f = low_f
            self.accel_filter_low_h = high_f

            for data_type in ["hip"]:

                # DATA SET UP

                if data_type == "hip" and self.hip_fname is not None:
                    data = np.array([self.df_hip["X"], self.df_hip["Y"], self.df_hip["Z"]])
                    original_df = self.df_hip
                    fs = self.hip_samplerate * .5

                # FILTERING TYPES
                if filter_type == "lowpass":
                    print("\nFiltering {} accelerometer data with {}Hz, "
                          "order {} lowpass filter.".format(data_type, low_f, filter_order))
                    low = low_f / fs
                    b, a = butter(N=filter_order, Wn=low, btype="lowpass")
                    filtered_data = filtfilt(b, a, x=data)

                    self.filter_details = {"Order": filter_order, "Type": filter_type, "F crit": [low_f]}

                if filter_type == "highpass":
                    print("\nFiltering {} accelerometer data with {}Hz, "
                          "order {} highpass filter.".format(data_type, high_f, filter_order))
                    high = high_f / fs
                    b, a = butter(N=filter_order, Wn=high, btype="highpass")
                    filtered_data = filtfilt(b, a, x=data)

                    self.filter_details = {"Order": filter_order, "Type": filter_type, "F crit": [high_f]}

                if filter_type == "bandpass":
                    print("\nFiltering {} accelerometer data with {}-{}Hz, "
                          "order {} bandpass filter.".format(data_type, low_f, high_f, filter_order))

                    low = low_f / fs
                    high = high_f / fs
                    b, a = butter(N=filter_order, Wn=[low, high], btype="bandpass")
                    filtered_data = filtfilt(b, a, x=data)

                    self.filter_details = {"Order": filter_order, "Type": filter_type, "F crit": [low_f, high_f]}

                original_df["X_filt"] = filtered_data[0]
                original_df["Y_filt"] = filtered_data[1]
                original_df["Z_filt"] = filtered_data[2]

        print("\nFiltering complete.")

    # ==================================================== BLOCK 2C ===================================================

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

        # Window length in minutes
        window_len = (stop_stamp - start_stamp).total_seconds() / 60

        print("Plotting {} minute section from {} to {}.".format(round(window_len, 2),
                                                                 datetime.strftime(start_stamp, "%Y-%m-%d %H:%M:%S"),
                                                                 datetime.strftime(stop_stamp, "%Y-%m-%d %H:%M:%S")))

        # Downsampling information ------------------------------------------------------------------------------------
        if downsample_factor != 1:
            if self.hip_fname is not None:
                print("\nDownsampling {}Hz data by a factor of {}. "
                      "New data is {}Hz.".format(self.hip_samplerate, downsample_factor,
                                                 round(self.hip_samplerate / downsample_factor, 1)))

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
        if 1/12 < window_len <= .5:
            locator = mdates.SecondLocator(interval=2)

        # Plots ~15 ticks if window less than 5 seconds
        if window_len <= 1/12:
            locator = mdates.MicrosecondLocator(interval=int(1000000 * (window_len*60/15)))

        # Plots depending on what data is available -------------------------------------------------------------------
        if self.hip_fname is not None:

            def plot_hip():
                fig, ax1 = plt.subplots(1, figsize=(self.fig_width, self.fig_height))
                plt.subplots_adjust(bottom=bottom_plot_crop_value)

                ax1.set_title("{} ({} Hz)".format(self.hip_fname.split("/")[-1],
                                                  int(self.hip_samplerate/downsample_factor)))
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

        f_name = self.check_file_overwrite("Hip_{}Hz_{} to {}".format(int(self.hip_samplerate/downsample_factor),
                                                                      datetime.strftime(start_stamp,
                                                                                        "%Y-%m-%d %H_%M_%S"),
                                                                      datetime.strftime(stop_stamp,
                                                                                        "%Y-%m-%d %H_%M_%S")))
        plt.savefig(f_name + ".png")
        print("Plot saved as png ({}.png)".format(f_name))

    # ==================================================== BLOCK 2D ===================================================

    def compare_filter(self, device_type=None, start=None, stop=None, downsample_factor=1):
        """Plots raw and filtered data on separate subplots.
        arguments:
            -device_type: "hip" --> which device to plot
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

            if device_type == "hip" or device_type == "Hip":
                print("\nDownsampling {}Hz data by a factor of {}. "
                      "New data is {}Hz.".format(self.hip_samplerate, downsample_factor,
                                                 round(self.hip_samplerate / downsample_factor, 1)))

        # Window length in minutes
        try:
            window_len = (stop - start).seconds / 60
        except TypeError:
            window_len = (datetime.strptime(stop, "%Y-%m-%d %H:%M:%S") -
                          datetime.strptime(start, "%Y-%m-%d %H:%M:%S")).seconds / 60

        if data_type == "absolute":
            df["Timestamp"] = np.arange(0, (stop - start).seconds, 1 / fs)[0:df.shape[0]]

        print("Plotting {} data: {} minute section from "
              "{} to {}.".format(device_type, round(window_len, 1),
                                 datetime.strftime(self.start_stamp, "%Y-%m-%d %H:%M:%S"),
                                 datetime.strftime(self.stop_stamp, "%Y-%m-%d %H:%M:%S")))

        # Formatting x-axis ticks ------------------------------------------------------------------------------------
        xfmt = mdates.DateFormatter("%a %b %d, %H:%M:%S")

        # Generates ~15 ticks (1/15th of window length apart)
        locator = mdates.MinuteLocator(byminute=np.arange(0, 59, int(np.ceil(window_len / 15))), interval=1)

        # ACCELEROMETER PLOTTING -------------------------------------------------------------------------------------
        if device_type != "ECG" and device_type != "ecg":
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(self.fig_width, self.fig_height))
            plt.subplots_adjust(bottom=.17, hspace=.33)

            ax1.set_title("Unfiltered {} data".format(device_type))
            ax1.plot(df["Timestamp"], df["X"], label="X", color='red')
            ax1.plot(df["Timestamp"], df["Y"], label="Y", color='black')
            ax1.plot(df["Timestamp"], df["Z"], label="Z", color='dodgerblue')
            ax1.set_ylabel("G's")
            ax1.legend(loc='lower left')

            if self.filter_details["Type"] == "bandpass":
                ax2.set_title("Filtered {} data (order {}, "
                              "{}-{} Hz bandpass)".format(device_type, self.filter_details["Order"],
                                                          self.filter_details["F crit"][0],
                                                          self.filter_details["F crit"][1]))
                filename = "{}_Raw_and_{}to{}HzBandpassFilter_" \
                           "{} to {}.png".format(device_type.capitalize(),
                                                 self.filter_details["F crit"][0],
                                                 self.filter_details["F crit"][1],
                                                 datetime.strftime(start, "%Y-%m-%d %H_%M_%S"),
                                                 datetime.strftime(stop, "%Y-%m-%d %H_%M_%S"))

            if self.filter_details["Type"] == "lowpass" or self.filter_details["Type"] == "highpass":
                ax2.set_title("Filtered {} data (order {}, {} Hz {})".format(device_type,
                                                                             self.filter_details["Order"],
                                                                             self.filter_details["F crit"][0],
                                                                             self.filter_details["Type"]))

                filename = "{}_Raw_and_{}Hz{}Filter_" \
                           "{} to {}.png".format(device_type.capitalize(),
                                                 self.filter_details["F crit"][0],
                                                 self.filter_details["Type"].capitalize(),
                                                 datetime.strftime(start, "%Y-%m-%d %H_%M_%S"),
                                                 datetime.strftime(stop, "%Y-%m-%d %H_%M_%S"))

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

        plt.xticks(rotation=45, fontsize=8)

        f_name = self.check_file_overwrite(filename)
        plt.savefig(f_name)
        print("Plot saved as png ({})".format(f_name))

    # ==================================================== BLOCK 2E ===================================================

    def get_timestamps(self, start=None, stop=None):

        # If start/stop given as integer, sets start/stop stamps to minutes into collection ---------------------------
        if (type(start) is int or type(start) is float) and (type(stop) is int or type(stop) is float):

            data_type = "absolute"

            start_stamp = self.df_hip["Timestamp"].iloc[0] + timedelta(minutes=start)
            stop_stamp = self.df_hip["Timestamp"].iloc[0] + timedelta(minutes=stop)

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
            # Sets start/stop to first/last timestamp for hip or wrist data
            if start is None and self.start_stamp is None:
                start_stamp = self.df_hip["Timestamp"].iloc[0]

            if stop is None and self.stop_stamp is None:
                stop_stamp = self.df_hip["Timestamp"].iloc[-1]

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

    @ staticmethod
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
            return filename.split(".")[0] + "_Version{}".format(version)
        if not file_exists:
            return filename


# ==================================================== BLOCK 3 =====================================================

# x = Wearables(hip_filepath="/Users/atrinh/Programming/Python/KIN471-Lab-2a/Hip.csv")
x = Wearables(hip_filepath="Hip_2a.csv")

# ==================================================== BLOCK 4 =====================================================

print("-Hip sampling rate = {} Hz".format(x.hip_samplerate))  # displays the sampling frequency (Hz)
print(x.df_hip[["X", "Y", "Z"]])  # prints out the first few rows of raw data

# ADDITIONAL FUNCTIONS TO RUN -----------------------------------------------------------------------------------------

# ===================================================== BLOCK 5A =====================================================

x.filter_signal(device_type="accelerometer", filter_type="bandpass", low_f=1, high_f=10, filter_order=3)

# ==================================================== BLOCK 6A =====================================================

# x.plot_data(downsample_factor=1)
# x.plot_data(start="2020-08-24 13:04:35", stop="2020-08-24 13:26:30", downsample_factor=1) # Main walk
# x.plot_data(start="2020-08-24 13:13:03", stop="2020-08-24 13:13:24", downsample_factor=1) # Single bout of walk
# x.plot_data(start="2020-08-24 13:13:08", stop="2020-08-24 13:13:12", downsample_factor=1) # 2 cycles of gait
x.plot_data(start="2020-08-24 13:13:09.5", stop="2020-08-24 13:13:10.5", downsample_factor=1)  # Stride cycle
# x.plot_data(start=15, stop=20, downsample_factor=1)

# ==================================================== BLOCK 7A =====================================================

# x.compare_filter(device_type="hip", downsample_factor=1)
# x.compare_filter(device_type="ECG", downsample_factor=1, start=0, stop=1)

# ==================================================== TIME RESET =====================================================

# x.start_stamp, x.stop_stamp = None, None

# ==================================================== HELP ===========================================================

# help(Wearables.compare_filter)

"""============================================= NEW IN THIS VERSION ==============================================="""
# formatted filenames
    # compare_filter() filenames now include filter details
        # done using Wearables.filter_details dictionary which contains details of last filter that was run
# Added Wearables.check_file_overwrite()
    # Method which adds "_VersionX" to end of file is a new file of the same name is written
    # Not really necessary since filter names are now included but may be useful in future