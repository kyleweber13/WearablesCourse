import pandas as pd  # package for data analysis and manipulation tools
from datetime import datetime  # module supplying classes for manipulating dates and times
import matplotlib.pyplot as plt  # library for creating static, animated, and interactive visualizations
import matplotlib.dates as mdates  # library for formatting plot axes labels as dates
import os  # module allowing code to use operating system dependent functionality
import matplotlib.ticker as ticker


class Wearables:

    def __init__(self, leftwrist_filepath=None, leftankle_filepath=None, rightankle_filepath=None,
                 fig_height=7, fig_width=12):

        # Default values for objects ----------------------------------------------------------------------------------
        self.lankle_fname = leftankle_filepath
        self.rankle_fname = rightankle_filepath
        self.wrist_fname = leftwrist_filepath

        self.fig_height = fig_height
        self.fig_width = fig_width

        self.epoched_df = None  # dataframe of all devices for one epoch length

        self.activity_volume = None  # activity volume for one epoch length

        self.df_all_volumes = None  # activity volumes for 1, 5, 15, 30, and 60-second epochs

        # Methods and objects that are run automatically when class instance is created -------------------------------

        self.df_lankle, self.lankle_samplerate = self.load_correct_file(filepath=self.lankle_fname,
                                                                        f_type="Left Ankle")
        self.df_rankle, self.rankle_samplerate = self.load_correct_file(filepath=self.rankle_fname,
                                                                        f_type="Right Ankle")
        self.df_wrist, self.wrist_samplerate = self.load_correct_file(filepath=self.wrist_fname,
                                                                      f_type="Left Wrist")

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

    def recalculate_epoch_len(self, epoch_len, print_statement=True, write_file=False):
        """Method to re-epoch the 1-second epoch files into any epoch length. Also scales cutpoints and stores these
           values in self.cutpoint_dict.
           Calls self.calculate_activity_volume(). Volumes are not printed but are stored in self.activity_volume df

        :arguments
        -epoch_len: desired epoch length in seconds
        -write_file: boolean of whether to save epoched_df as .csv
        """

        if print_statement:
            print("\nRecalculating epoch length to {} seconds...".format(epoch_len))

        lankle_epoched = [None for i in range(self.df_lankle.shape[0])]
        rankle_epoched = [None for i in range(self.df_rankle.shape[0])]
        wrist_epoched = [None for i in range(self.df_wrist.shape[0])]

        timestamps_found = False

        if self.df_lankle is not None:
            timestamps = self.df_lankle["Timestamp"].iloc[::epoch_len]
            timestamps_found = True
            df_timestamps = timestamps

            svm = [i for i in self.df_lankle["SVM"]]

            lankle_epoched = [sum(svm[i:i+epoch_len]) for i in range(0, self.df_lankle.shape[0], epoch_len)]

        if self.df_rankle is not None:
            timestamps = self.df_rankle["Timestamp"].iloc[::epoch_len]

            if not timestamps_found:
                df_timestamps = timestamps
                timestamps_found = True

            svm = [i for i in self.df_rankle["SVM"]]

            rankle_epoched = [sum(svm[i:i + epoch_len]) for i in range(0, self.df_rankle.shape[0], epoch_len)]

        if self.df_wrist is not None:
            timestamps = self.df_wrist["Timestamp"].iloc[::epoch_len]

            if not timestamps_found:
                df_timestamps = timestamps

            svm = [i for i in self.df_wrist["SVM"]]

            wrist_epoched = [sum(svm[i:i + epoch_len]) for i in range(0, self.df_wrist.shape[0], epoch_len)]

        # Combines all devices' counts into one dataframe
        self.epoched_df = pd.DataFrame(list(zip(df_timestamps, lankle_epoched, rankle_epoched, wrist_epoched)),
                                       columns=["Timestamp", "LAnkle", "RAnkle", "Wrist"])

        # Saves dataframe to csv
        if write_file:
            self.epoched_df.to_csv("Epoch{}_All.csv".format(epoch_len), index=False)

        # Scales cutpoints
        self.cutpoint_dict = {"Light": self.cutpoint_dict["Light"] *
                                       (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "Moderate": self.cutpoint_dict["Moderate"] *
                                          (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "Vigorous": self.cutpoint_dict["Vigorous"] *
                                          (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "Epoch length": epoch_len}

        # Tallies activity intensity totals
        self.calculate_activity_volume()

    def calculate_activity_volume(self):
        """Calculates activity volume for one self.epoched_df. Called by self.recalculate_epoch_len and
           self.calculate_all_activity_volumes() methods.
        """

        sed_epochs = self.epoched_df["Wrist"].loc[(self.epoched_df["Wrist"] < self.cutpoint_dict["Light"])].shape[0]

        light_epochs = self.epoched_df["Wrist"].loc[(self.epoched_df["Wrist"] >=
                                                     self.cutpoint_dict["Light"]) &
                                                    (self.epoched_df["Wrist"] <
                                                     self.cutpoint_dict["Moderate"])].shape[0]

        mod_epochs = self.epoched_df["Wrist"].loc[(self.epoched_df["Wrist"] >=
                                                   self.cutpoint_dict["Moderate"]) &
                                                  (self.epoched_df["Wrist"] <
                                                   self.cutpoint_dict["Vigorous"])].shape[0]

        vig_epochs = self.epoched_df["Wrist"].loc[(self.epoched_df["Wrist"] >=
                                                   self.cutpoint_dict["Vigorous"])].shape[0]

        activity_minutes = {"Sedentary": round(sed_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Light": round(light_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Moderate": round(mod_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Vigorous": round(vig_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2)}

        self.activity_volume = activity_minutes

    def plot_activity_counts(self):
        """Plots activity counts from all accelerometers from whatever last epoch length called in
        self.recalculate_epoch_len was. Requires that method was called at least once."""

        print("\nPlotting all {}-second epoch data...".format(self.cutpoint_dict["Epoch length"]))

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex="col", figsize=(self.fig_width, self.fig_height))
        plt.suptitle("Data scaled to {}-second epochs".format(self.cutpoint_dict["Epoch length"]))

        ax1.plot(self.epoched_df["Timestamp"], self.epoched_df["Wrist"], color='dodgerblue', label="Wrist")
        ax1.set_ylabel("Counts / {} seconds".format(self.cutpoint_dict["Epoch length"]))
        ax1.axhline(y=self.cutpoint_dict["Light"], color='green', linestyle='dashed', label="Light")
        ax1.axhline(y=self.cutpoint_dict["Moderate"], color='darkorange', linestyle='dashed', label="Mod.")
        ax1.axhline(y=self.cutpoint_dict["Vigorous"], color='red', linestyle='dashed', label="Vig.")

        ax1.legend()

        ax2.plot(self.epoched_df["Timestamp"], self.epoched_df["LAnkle"], color='red', label="LAnkle")
        ax2.set_ylabel("Counts / {} seconds".format(self.cutpoint_dict["Epoch length"]))
        ax2.legend()

        ax3.plot(self.epoched_df["Timestamp"], self.epoched_df["RAnkle"], color='black', label="RAnkle")
        ax3.set_ylabel("Counts / {} seconds".format(self.cutpoint_dict["Epoch length"]))
        ax3.legend()

        xfmt = mdates.DateFormatter("%H:%M:%S %p")
        ax3.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        f_name = self.check_file_overwrite("{}second_epochs".format(self.cutpoint_dict["Epoch length"]))
        plt.savefig(f_name)
        print("Plot saved as png ({})".format(f_name))

    def plot_counts_demo(self):
        """Method to plot 1, 15, and 60-second wrist epoch data as barplot to show
           differences in counts and cutpoints
        """

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(self.fig_width, self.fig_height))
        plt.suptitle("Demo: wrist activity counts during desk work (6-minute window)")

        for i, epoch in enumerate([1, 15, 60]):
            self.recalculate_epoch_len(epoch_len=epoch, print_statement=False)

            df = self.epoched_df.loc[(self.epoched_df["Timestamp"] >= "2020-10-08 9:31:00") &
                                     (self.epoched_df["Timestamp"] <= "2020-10-08 9:37:00")]

            if i == 0:
                ax1.set_title("1-second epochs")
                ax1.bar(df["Timestamp"], df["Wrist"], edgecolor='black', color='dodgerblue', align='edge',
                        width=self.cutpoint_dict["Epoch length"] / 86400)
                ax1.set_ylabel("Counts / {} seconds".format(self.cutpoint_dict["Epoch length"]))
                ax1.axhline(y=self.cutpoint_dict["Light"], color='green', linestyle='dashed', label="Light")
                ax1.axhline(y=self.cutpoint_dict["Moderate"], color='darkorange', linestyle='dashed', label="Mod.")
                ax1.axhline(y=self.cutpoint_dict["Vigorous"], color='red', linestyle='dashed', label="Vig.")
                ax1.legend()

            if i == 1:
                ax2.set_title("15-second epochs")
                ax2.bar(df["Timestamp"], df["Wrist"], edgecolor='black', color='dodgerblue', align='edge',
                        width=self.cutpoint_dict["Epoch length"] / 86400)
                ax2.set_ylabel("Counts / {} seconds".format(self.cutpoint_dict["Epoch length"]))
                ax2.axhline(y=self.cutpoint_dict["Light"], color='green', linestyle='dashed')
                ax2.axhline(y=self.cutpoint_dict["Moderate"], color='darkorange', linestyle='dashed')
                ax2.axhline(y=self.cutpoint_dict["Vigorous"], color='red', linestyle='dashed')

            if i == 2:
                ax3.set_title("60-second epochs")
                ax3.bar(df["Timestamp"], df["Wrist"], edgecolor='black', color='dodgerblue', align='edge',
                        width=self.cutpoint_dict["Epoch length"] / 86400)
                ax3.set_ylabel("Counts / {} seconds".format(self.cutpoint_dict["Epoch length"]))
                ax3.axhline(y=self.cutpoint_dict["Light"], color='green', linestyle='dashed')
                ax3.axhline(y=self.cutpoint_dict["Moderate"], color='darkorange', linestyle='dashed')
                ax3.axhline(y=self.cutpoint_dict["Vigorous"], color='red', linestyle='dashed')

                xfmt = mdates.DateFormatter("%H:%M:%S %p")
                ax3.xaxis.set_major_formatter(xfmt)
                plt.xticks(rotation=45, fontsize=8)

        plt.savefig("Wrist_ActivityCounts_Demo.png")
        print("\nSaved plot as Wrist_ActivityCounts_Demo.png")

    def calculate_all_activity_volumes(self, save_file=False, show_plot=True):
        """Calculates activity volumes for 1, 5, 15, 30, and 60-second epochs. Able to plot results as barplot and to
           save results as .csv

        :argument
        -save_file: boolean of whether to save results csv
        -save_plot: boolean to show barplot
        """

        print("\nCalculating activity volumes for 1, 5, 15, 30, and 60-second epochs...")

        volumes = []
        for epoch in [1, 5, 15, 30, 60]:
            self.recalculate_epoch_len(epoch_len=epoch, print_statement=False)
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

            f_name = self.check_file_overwrite("ActivityVolumes_by_EpochLength")
            plt.savefig(f_name)
            print("Plot saved as png ({})".format(f_name))


x = Wearables(leftwrist_filepath="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 5/LWrist_Epoch1.csv",
              leftankle_filepath="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 5/LAnkle_Epoch1.csv",
              rightankle_filepath="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 5/RAnkle_Epoch1.csv")

# Bar plot to show how counts from different epoch compare. Shows scaled cutpoints.
# x.plot_counts_demo()

# Able to pick any epoch length (seconds). To save as csv: write_file=True
# x.recalculate_epoch_len(epoch_len=15, write_file=True)

# Plots most recent epoch length calculation. Saves plot.
# x.plot_activity_counts()

# Calculates activity volume for 1, 5, 15, 30, and 60-second epochs. Saves to csv if save_file=True.
# x.calculate_all_activity_volumes(save_file=False, show_plot=True)
