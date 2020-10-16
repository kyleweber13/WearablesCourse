import pandas as pd  # package for data analysis and manipulation tools
from datetime import datetime  # module supplying classes for manipulating dates and times
import matplotlib.pyplot as plt  # library for creating static, animated, and interactive visualizations
import matplotlib.dates as mdates  # library for formatting plot axes labels as dates
import os  # module allowing code to use operating system dependent functionality
from datetime import timedelta


class Wearables:

    def __init__(self, leftwrist_filepath=None, leftankle_filepath=None, fig_height=7, fig_width=12):
        """Class that reads in one wrist and one ankle csv file that has been epoched into 1-second epochs.
           Able to re-epoch into different epoch lengths and calculate activity volumes using scaled cutpoints.
        """

        # Default values for objects ----------------------------------------------------------------------------------
        self.ankle_fname = leftankle_filepath
        self.wrist_fname = leftwrist_filepath

        self.fig_height = fig_height
        self.fig_width = fig_width

        self.start_stamp = None
        self.stop_stamp = None

        self.epoched_df = None  # dataframe of all devices for one epoch length

        self.activity_volume = None  # activity volume for one epoch length

        self.df_all_volumes = None  # activity volumes for 1, 5, 15, 30, and 60-second epochs

        # Methods and objects that are run automatically when class instance is created -------------------------------

        self.df_ankle, self.ankle_samplerate = self.load_correct_file(filepath=self.ankle_fname,
                                                                      f_type="Left Ankle")
        self.df_wrist, self.wrist_samplerate = self.load_correct_file(filepath=self.wrist_fname,
                                                                      f_type="Left Wrist")

        # Crops dataframes so length is a multiple of 1, 5, 10, 15, 30, and 60-second epochs
        self.crop_dataframes()

        # Powell et al., 2016 cutpoints scaled to 75 Hz sampling rate and 1-second epoch
        # Original is 30 Hz and 15-second epochs
        self.cutpoint_dict = {"Light": round(47 * self.wrist_samplerate / 30 / 15, 2),
                              "Moderate": round(64 * self.wrist_samplerate / 30 / 15, 2),
                              "Vigorous": round(157 * self.wrist_samplerate / 30 / 15, 2),
                              "Epoch length": 1}

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

    def crop_dataframes(self):

        file_dur = self.df_wrist.__len__()

        crop_ind = file_dur - file_dur % 60

        self.df_wrist = self.df_wrist.iloc[:crop_ind]
        self.df_ankle = self.df_ankle.iloc[:crop_ind]

        print("\nCropped {} 1-second epochs off of data so that data length is a "
              "multiple of all potential epoch lengths.".format(int(file_dur % 60)))

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

        # If start/stop given as integer, sets start/stop stamps to minutes into collection ---------------------------
        if (type(start) is int or type(start) is float) and (type(stop) is int or type(stop) is float):

            data_type = "absolute"

            start_stamp = self.df_ankle["Timestamp"].iloc[0] + timedelta(minutes=start)
            stop_stamp = self.df_ankle["Timestamp"].iloc[0] + timedelta(minutes=stop)

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
                start_stamp = self.df_ankle["Timestamp"].iloc[0]

            if stop is None and self.stop_stamp is None:
                stop_stamp = self.df_ankle["Timestamp"].iloc[-1]

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

    def recalculate_epoch_len(self, epoch_len, print_statement=True, write_file=False, calculate_volume=True,
                              start=None, stop=None):
        """Method to re-epoch the 1-second epoch files into any epoch length. Also scales cutpoints and stores these
           values in self.cutpoint_dict.
           Calls self.calculate_activity_volume(). Volumes are not printed but are stored in self.activity_volume df
           Able to crop data using start/stop timestamps.

        :arguments
        -epoch_len: desired epoch length in seconds
        -write_file: boolean of whether to save epoched_df as .csv
        -start/stop: timestamp in format YYYY-mm-dd HH:MM:SS
        """

        if print_statement:
            print("\nRecalculating epoch length to {} seconds...".format(epoch_len))

        if start is not None and stop is not None:
            df_ankle = self.df_ankle.loc[(self.df_ankle["Timestamp"] >= start) &
                                         (self.df_ankle["Timestamp"] < stop)]
            df_wrist = self.df_wrist.loc[(self.df_wrist["Timestamp"] >= start) &
                                         (self.df_wrist["Timestamp"] < stop)]

        if start is None and stop is None:
            df_ankle = self.df_ankle
            df_wrist = self.df_wrist

        ankle_epoched = [None for i in range(df_ankle.shape[0])]
        wrist_epoched = [None for i in range(df_wrist.shape[0])]

        timestamps_found = False

        if df_ankle is not None:
            timestamps = df_ankle["Timestamp"].iloc[::epoch_len]

            timestamps_found = True
            df_timestamps = timestamps

            svm = [i for i in df_ankle["SVM"]]

            ankle_epoched = [sum(svm[i:i+epoch_len]) for i in range(0, df_ankle.shape[0], epoch_len)]

        if self.df_wrist is not None:
            timestamps = df_wrist["Timestamp"].iloc[::epoch_len]

            if not timestamps_found:
                df_timestamps = timestamps

            svm = [i for i in df_wrist["SVM"]]

            wrist_epoched = [sum(svm[i:i + epoch_len]) for i in range(0, self.df_wrist.shape[0], epoch_len)]

        # Combines all devices' counts into one dataframe
        self.epoched_df = pd.DataFrame(list(zip(df_timestamps, ankle_epoched, wrist_epoched)),
                                       columns=["Timestamp", "LAnkle", "LWrist"])

        # Saves dataframe to csv
        if write_file:
            self.epoched_df.to_csv("Epoch{}_All_{}_to_{}.csv".format(epoch_len, start, stop), index=False)

        # Scales cutpoints
        self.cutpoint_dict = {"Light": self.cutpoint_dict["Light"] *
                                       (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "Moderate": self.cutpoint_dict["Moderate"] *
                                          (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "Vigorous": self.cutpoint_dict["Vigorous"] *
                                          (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "Epoch length": epoch_len}

        # Tallies activity intensity totals
        if calculate_volume:
            self.calculate_activity_volume(start=start, stop=stop)

    def calculate_activity_volume(self, start=None, stop=None):
        """Calculates activity volume for one self.epoched_df. Called by self.recalculate_epoch_len and
           self.calculate_all_activity_volumes() methods.
           Able to crop data using start/stop.

           :argument
           -start/stop: timestamp in format YYYY-mm-dd HH:MM:SS
        """

        if self.epoched_df is None:
            self.epoched_df = pd.DataFrame(list(zip(self.df_wrist["Timestamp"],
                                                    self.df_ankle["SVM"],
                                                    self.df_wrist["SVM"])),
                                           columns=["Timestamp", "LAnkle", "LWrist"])

        if start is not None and stop is None:
            stop = self.epoched_df.iloc[-1]["Timestamp"]

        if start is not None and stop is not None:
            df = self.epoched_df.loc[(self.epoched_df["Timestamp"] >= start) &
                                     (self.epoched_df["Timestamp"] <= stop)]

        if start is None and stop is None:
            df = self.epoched_df

        print("\nCalculating activity data from {} to {} "
              "in {}-second epochs...".format(df.iloc[0]["Timestamp"],
                                              df.iloc[-1]['Timestamp'],
                                              self.cutpoint_dict["Epoch length"]))

        sed_epochs = df["LWrist"].loc[(df["LWrist"] < self.cutpoint_dict["Light"])].shape[0]

        light_epochs = df["LWrist"].loc[(df["LWrist"] >= self.cutpoint_dict["Light"]) &
                                        (df["LWrist"] < self.cutpoint_dict["Moderate"])].shape[0]

        mod_epochs = df["LWrist"].loc[(df["LWrist"] >= self.cutpoint_dict["Moderate"]) &
                                      (df["LWrist"] < self.cutpoint_dict["Vigorous"])].shape[0]

        vig_epochs = df["LWrist"].loc[(df["LWrist"] >= self.cutpoint_dict["Vigorous"])].shape[0]

        activity_minutes = {"Sedentary": round(sed_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Light": round(light_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Moderate": round(mod_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Vigorous": round(vig_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2)}

        self.activity_volume = activity_minutes
        print("Activity volume:")
        print(self.activity_volume)

    def plot_activity_counts(self, start=None, stop=None, epoch_len=15):
        """Plots activity counts from both accelerometers. Saves plot.

           :argument
           -start/stop: timestamp in format YYYY-mm-dd HH:MM:SS
           -epoch_len: epoch length in seconds
        """

        print("\nPlotting all {}-second epoch data...".format(self.cutpoint_dict["Epoch length"]))

        if epoch_len != self.cutpoint_dict["Epoch length"]:
            self.recalculate_epoch_len(epoch_len=epoch_len, print_statement=True)

        if self.epoched_df is None:
            self.epoched_df = pd.DataFrame(list(zip(self.df_wrist["Timestamp"],
                                                    self.df_ankle["SVM"],
                                                    self.df_wrist["SVM"])),
                                           columns=["Timestamp", "LAnkle", "LWrist"])

        if start is None and stop is None:
            start = self.epoched_df.iloc[0]["Timestamp"]
            stop = self.epoched_df.iloc[-1]["Timestamp"]

        df = self.epoched_df.loc[(self.epoched_df["Timestamp"] > start) &
                                 (self.epoched_df["Timestamp"] < stop)]

        fig, (ax1, ax2) = plt.subplots(2, sharex="col", figsize=(self.fig_width, self.fig_height))
        plt.suptitle("Data scaled to {}-second epochs".format(self.cutpoint_dict["Epoch length"]))

        ax1.plot(df["Timestamp"], df["LWrist"], color='dodgerblue', label="Wrist")
        ax1.set_ylabel("Counts / {} seconds".format(self.cutpoint_dict["Epoch length"]))
        ax1.axhline(y=self.cutpoint_dict["Light"], color='green', linestyle='dashed', label="Light")
        ax1.axhline(y=self.cutpoint_dict["Moderate"], color='darkorange', linestyle='dashed', label="Mod.")
        ax1.axhline(y=self.cutpoint_dict["Vigorous"], color='red', linestyle='dashed', label="Vig.")

        ax1.legend()

        ax2.plot(df["Timestamp"], df["LAnkle"], color='red', label="LAnkle")
        ax2.set_ylabel("Counts / {} seconds".format(self.cutpoint_dict["Epoch length"]))
        ax2.legend()

        xfmt = mdates.DateFormatter("%H:%M:%S %p")
        ax2.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        f_name = self.check_file_overwrite("{}second_epochs_{} "
                                           "to {}".format(self.cutpoint_dict["Epoch length"],
                                                          datetime.strftime(start, "%Y-%m-%d %H_%M_%S"),
                                                          datetime.strftime(stop, "%Y-%m-%d %H_%M_%S")))
        plt.savefig(f_name)
        print("Plot saved as png ({})".format(f_name))

    def plot_different_epoch_lengths(self, epoch_lens=(1, 15, 60), accel_location="Wrist"):
        """Method to plot wrist or ankle epoch data in 3 epoch lengths as barplot to show
           differences in counts and cutpoints (wrist only). Saves plot.

            :argument
            -epoch_lens: list of length 3 specifying epoch lengths. Keep to 60 seconds max.
            -accel_location: "Wrist" or "Ankle"; which data to plot
        """

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(self.fig_width, self.fig_height))
        plt.suptitle("Demo: wrist activity counts during desk work (6-minute window)")

        for i, epoch in enumerate(epoch_lens):
            self.recalculate_epoch_len(epoch_len=epoch, print_statement=False, calculate_volume=False)

            df = self.epoched_df.loc[(self.epoched_df["Timestamp"] >= "2020-10-08 9:31:00") &
                                     (self.epoched_df["Timestamp"] <= "2020-10-08 9:37:00")]

            if accel_location.capitalize() == "Wrist":
                data = df["LWrist"]
            if accel_location.capitalize() == "Ankle":
                data = df["LAnkle"]

            if i == 0:
                ax1.set_title("{}-second epochs".format(epoch))
                ax1.bar(df["Timestamp"], data, edgecolor='black', color='dodgerblue', align='edge',
                        width=self.cutpoint_dict["Epoch length"] / 86400)
                ax1.set_ylabel("Counts / {} seconds".format(self.cutpoint_dict["Epoch length"]))

                if accel_location.capitalize() == "Wrist":
                    ax1.axhline(y=self.cutpoint_dict["Light"], color='green', linestyle='dashed', label="Light")
                    ax1.axhline(y=self.cutpoint_dict["Moderate"], color='darkorange', linestyle='dashed', label="Mod.")
                    ax1.axhline(y=self.cutpoint_dict["Vigorous"], color='red', linestyle='dashed', label="Vig.")

                    ax1.legend()

            if i == 1:
                ax2.set_title("{}-second epochs".format(epoch))
                ax2.bar(df["Timestamp"], data, edgecolor='black', color='dodgerblue', align='edge',
                        width=self.cutpoint_dict["Epoch length"] / 86400)
                ax2.set_ylabel("Counts / {} seconds".format(self.cutpoint_dict["Epoch length"]))

                if accel_location.capitalize() == "Wrist":
                    ax2.axhline(y=self.cutpoint_dict["Light"], color='green', linestyle='dashed')
                    ax2.axhline(y=self.cutpoint_dict["Moderate"], color='darkorange', linestyle='dashed')
                    ax2.axhline(y=self.cutpoint_dict["Vigorous"], color='red', linestyle='dashed')

            if i == 2:
                ax3.set_title("{}-second epochs".format(epoch))
                ax3.bar(df["Timestamp"], data, edgecolor='black', color='dodgerblue', align='edge',
                        width=self.cutpoint_dict["Epoch length"] / 86400)
                ax3.set_ylabel("Counts / {} seconds".format(self.cutpoint_dict["Epoch length"]))

                if accel_location.capitalize() == "Wrist":
                    ax3.axhline(y=self.cutpoint_dict["Light"], color='green', linestyle='dashed')
                    ax3.axhline(y=self.cutpoint_dict["Moderate"], color='darkorange', linestyle='dashed')
                    ax3.axhline(y=self.cutpoint_dict["Vigorous"], color='red', linestyle='dashed')

                xfmt = mdates.DateFormatter("%H:%M:%S %p")
                ax3.xaxis.set_major_formatter(xfmt)
                plt.xticks(rotation=45, fontsize=8)

        plt.savefig("{}_ActivityCounts_Demo.png".format(accel_location))
        print("\nSaved plot as {}_ActivityCounts_Demo.png".format(accel_location))

    def calculate_all_activity_volumes(self, save_file=False, show_plot=True, start=None, stop=None):
        """Calculates activity volumes for 1, 5, 15, 30, and 60-second epochs. Able to plot results as barplot and to
           save results as .csv

        :argument
        -save_file: boolean of whether to save results csv
        -save_plot: boolean to show barplot
        """

        print("\nCalculating activity volumes for 1, 5, 15, 30, and 60-second epochs...")

        volumes = []
        for epoch in [1, 5, 15, 30, 60]:
            self.recalculate_epoch_len(epoch_len=epoch, print_statement=False, start=start, stop=stop)
            volumes.append([i for i in self.activity_volume.values()])

        df_volume = pd.DataFrame(volumes)
        df_volume["Epoch length"] = ["1s", "5s", "15s", "30s", "60s"]
        df_volume = df_volume.set_index("Epoch length", drop=True)
        df_volume.columns = ["Sedentary", "Light", "Moderate", "Vigorous"]
        df_volume["MVPA"] = df_volume["Moderate"] + df_volume["Vigorous"]

        self.df_all_volumes = df_volume
        print(self.df_all_volumes)

        if save_file:
            self.df_all_volumes.to_csv("All_activity_volumes.csv")

        if show_plot:
            plt.subplots(2, 2, figsize=(self.fig_width, self.fig_height))
            plt.suptitle("Activity volumes for 1, 5, 15, 30, and 60-second epochs")

            # Sedentary ----------------------------------------------------------------------------------------------
            plt.subplot(2, 2, 1)

            plt.title("Sedentary")
            plt.bar([i for i in self.df_all_volumes.index], [i for i in self.df_all_volumes["Sedentary"]],
                    edgecolor='black', color=["black", "dimgray", "gray", "darkgray", "lightgray"])
            plt.ylabel("Minutes")

            # Light ---------------------------------------------------------------------------------------------------
            plt.subplot(2, 2, 2)

            plt.title("Light")
            plt.bar([i for i in self.df_all_volumes.index], [i for i in self.df_all_volumes["Light"]],
                    edgecolor='black', color=["darkgreen", "green", "forestgreen", "limegreen", "lime"])

            # Moderate ------------------------------------------------------------------------------------------------
            plt.subplot(2, 2, 3)

            plt.title("Moderate")
            plt.bar([i for i in self.df_all_volumes.index], [i for i in self.df_all_volumes["Moderate"]],
                    edgecolor='black', color=["saddlebrown", "sienna", "chocolate", "darkorange", "orange"])
            plt.ylabel("Minutes")
            plt.xlabel("Epoch length")

            # Vigorous ------------------------------------------------------------------------------------------------
            plt.subplot(2, 2, 4)

            plt.title("Vigorous")
            plt.bar([i for i in self.df_all_volumes.index], [i for i in self.df_all_volumes["Vigorous"]],
                    edgecolor='black', color=["maroon", "darkred", "firebrick", "red", "tomato"])
            plt.xlabel("Epoch length")

            if start is not None and stop is not None:
                f_name = self.check_file_overwrite("ActivityVolumes_by_EpochLength_{}_to_{}".format(start, stop))
            if start is None and stop is None:
                f_name = self.check_file_overwrite("ActivityVolumes_by_EpochLength")

            plt.savefig(f_name)
            print("Plot saved as png ({})".format(f_name))


x = Wearables(leftwrist_filepath="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 5/LWrist_Epoch1.csv",
              leftankle_filepath="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 5/LAnkle_Epoch1.csv")

# Bar plot to show how counts from different epoch compare. Shows scaled cutpoints for wrist.
# Pick data using accel_location="Wrist" or "Ankle". Able to pick 3 epoch lengths (epoch_lens=[a, b, c])
# x.plot_different_epoch_lengths(epoch_lens=[1, 10, 60], accel_location="Wrist")

# Able to pick any epoch length (seconds). To save as csv: write_file=True
# x.recalculate_epoch_len(epoch_len=15, write_file=False)

# Calculates activity volume using wrist cutpoints of current epoch length
# x.calculate_activity_volume(start="2020-10-08 12:08:00", stop="2020-10-08 12:35:00")

# Plots most recent epoch length calculation. Able to crop using start/stop timestamps. Saves plot.
# x.plot_activity_counts(start=None, stop=None, epoch_len=15)

# Calculates activity volume for 1, 5, 15, 30, and 60-second epochs. Saves to csv if save_file=True.
# x.calculate_all_activity_volumes(save_file=True, show_plot=True, start="2020-10-08 12:09:00", stop="2020-10-08 12:35:00")
