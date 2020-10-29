import os
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt  # library for creating static, animated, and interactive visualizations
import matplotlib.dates as mdates  # library for formatting plot axes labels as dates


class Data:

    def __init__(self, leftwrist_filepath=None, rightwrist_filepath=None,
                 leftankle_filepath=None, rightankle_filepath=None, fig_height=7, fig_width=12):

        # Default values for objects ----------------------------------------------------------------------------------
        self.lankle_fname = leftankle_filepath
        self.lwrist_fname = leftwrist_filepath
        self.rankle_fname = rightankle_filepath
        self.rwrist_fname = rightwrist_filepath

        self.fig_height = fig_height
        self.fig_width = fig_width

        self.epoched_df = None  # dataframe of all devices for one epoch length

        self.activity_volume = None  # activity volume for one epoch length
        self.activity_df = None

        # Methods and objects that are run automatically when class instance is created -------------------------------

        self.df_lankle, self.lankle_samplerate = self.load_correct_file(filepath=self.lankle_fname,
                                                                        f_type="Left Ankle")
        self.df_lwrist, self.lwrist_samplerate = self.load_correct_file(filepath=self.lwrist_fname,
                                                                        f_type="Left Wrist")
        self.df_rankle, self.rankle_samplerate = self.load_correct_file(filepath=self.rankle_fname,
                                                                        f_type="Right Ankle")
        self.df_rwrist, self.rwrist_samplerate = self.load_correct_file(filepath=self.rwrist_fname,
                                                                        f_type="Right Wrist")

        # Powell et al., 2016 cutpoints scaled to 75 Hz sampling rate and 1-second epoch
        # Original is 30 Hz and 15-second epochs
        self.cutpoint_dict = {"NonDomLight": round(47 * self.lwrist_samplerate / 30 / 15, 2),
                              "NonDomModerate": round(64 * self.lwrist_samplerate / 30 / 15, 2),
                              "NonDomVigorous": round(157 * self.lwrist_samplerate / 30 / 15, 2),
                              "DomLight": round(51 * self.rwrist_samplerate / 30 / 15, 2),
                              "DomModerate": round(68 * self.rwrist_samplerate / 30 / 15, 2),
                              "DomVigorous": round(142 * self.rwrist_samplerate / 30 / 15, 2),
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
        -epoch_len: desired epoch length in seconds, int
        -write_file: boolean of whether to save epoched_df as .csv
        -start/stop: timestamp in format YYYY-mm-dd HH:MM:SS
        """

        if print_statement:
            print("\nRecalculating epoch length to {} seconds...".format(epoch_len))

        # Cropping data frames
        if start is not None and stop is not None:
            df_lankle = self.df_lankle.loc[(self.df_lankle["Timestamp"] >= start) &
                                         (self.df_lankle["Timestamp"] < stop)]
            df_lwrist = self.df_lwrist.loc[(self.df_lwrist["Timestamp"] >= start) &
                                         (self.df_lwrist["Timestamp"] < stop)]
            df_rankle = self.df_rankle.loc[(self.df_rankle["Timestamp"] >= start) &
                                         (self.df_rankle["Timestamp"] < stop)]
            df_rwrist = self.df_rwrist.loc[(self.df_rwrist["Timestamp"] >= start) &
                                         (self.df_rwrist["Timestamp"] < stop)]

        # Whole data frames if no cropping
        if start is None and stop is None:
            df_lankle = self.df_lankle
            df_lwrist = self.df_lwrist
            df_rankle = self.df_rankle
            df_rwrist = self.df_rwrist

        # Empty lists as placeholders for missing data
        lankle_epoched = [None for i in range(df_lankle.shape[0])]
        lwrist_epoched = [None for i in range(df_lwrist.shape[0])]
        rankle_epoched = [None for i in range(df_rankle.shape[0])]
        rwrist_epoched = [None for i in range(df_rwrist.shape[0])]

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

        if self.df_rankle is not None:
            timestamps = df_rankle["Timestamp"].iloc[::epoch_len]

            timestamps_found = True
            df_timestamps = timestamps

            svm = [i for i in df_rankle["SVM"]]

            rankle_epoched = [sum(svm[i:i + epoch_len]) for i in range(0, df_rankle.shape[0], epoch_len)]

        if self.df_rwrist is not None:
            timestamps = df_rwrist["Timestamp"].iloc[::epoch_len]

            if not timestamps_found:
                df_timestamps = timestamps

            svm = [i for i in df_rwrist["SVM"]]

            rwrist_epoched = [sum(svm[i:i + epoch_len]) for i in range(0, self.df_rwrist.shape[0], epoch_len)]

        # Combines all devices' counts into one dataframe
        self.epoched_df = pd.DataFrame(list(zip(df_timestamps, lankle_epoched, rankle_epoched,
                                                lwrist_epoched, rwrist_epoched)),
                                       columns=["Timestamp", "LAnkle", "RAnkle", "LWrist", "RWrist"])

        # Saves dataframe to csv
        if write_file:
            self.epoched_df.to_csv("Epoch{}_All_{}_to_{}.csv".format(epoch_len, start, stop), index=False)

        # Scales cutpoints
        self.cutpoint_dict = {"NonDomLight": self.cutpoint_dict["NonDomLight"] *
                                             (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "NonDomModerate": self.cutpoint_dict["NonDomModerate"] *
                                                (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "NonDomVigorous": self.cutpoint_dict["NonDomVigorous"] *
                                                (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "DomLight": self.cutpoint_dict["DomLight"] *
                                             (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "DomModerate": self.cutpoint_dict["DomModerate"] *
                                                (epoch_len / self.cutpoint_dict["Epoch length"]),
                              "DomVigorous": self.cutpoint_dict["DomVigorous"] *
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
            self.epoched_df = pd.DataFrame(list(zip(self.df_lwrist["Timestamp"],
                                                    self.df_lankle["SVM"],
                                                    self.df_rankle["SVM"],
                                                    self.df_lwrist["SVM"],
                                                    self.df_rwrist["SVM"])),
                                           columns=["Timestamp", "LAnkle", "RAnkle", "LWrist", "RWrist"])

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

        # Non-dominant (left) wrist -----------------------------------------------------------------------------------
        nd_sed_epochs = df["LWrist"].loc[(df["LWrist"] < self.cutpoint_dict["NonDomLight"])].shape[0]

        nd_light_epochs = df["LWrist"].loc[(df["LWrist"] >= self.cutpoint_dict["NonDomLight"]) &
                                        (df["LWrist"] < self.cutpoint_dict["NonDomModerate"])].shape[0]

        nd_mod_epochs = df["LWrist"].loc[(df["LWrist"] >= self.cutpoint_dict["NonDomModerate"]) &
                                      (df["LWrist"] < self.cutpoint_dict["NonDomVigorous"])].shape[0]

        nd_vig_epochs = df["LWrist"].loc[(df["LWrist"] >= self.cutpoint_dict["NonDomVigorous"])].shape[0]

        # Dominant (right) wrist --------------------------------------------------------------------------------------
        d_sed_epochs = df["LWrist"].loc[(df["LWrist"] < self.cutpoint_dict["DomLight"])].shape[0]

        d_light_epochs = df["LWrist"].loc[(df["LWrist"] >= self.cutpoint_dict["DomLight"]) &
                                        (df["LWrist"] < self.cutpoint_dict["DomModerate"])].shape[0]

        d_mod_epochs = df["LWrist"].loc[(df["LWrist"] >= self.cutpoint_dict["DomModerate"]) &
                                      (df["LWrist"] < self.cutpoint_dict["DomVigorous"])].shape[0]

        d_vig_epochs = df["LWrist"].loc[(df["LWrist"] >= self.cutpoint_dict["DomVigorous"])].shape[0]

        # Data storage ------------------------------------------------------------------------------------------------
        activity_minutes = {"Sedentary_ND": round(nd_sed_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Sedentary_Dom": round(d_sed_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Light_ND": round(nd_light_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Light_Dom": round(d_light_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Moderate_ND": round(nd_mod_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Moderate_Dom": round(d_mod_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Vigorous_ND": round(nd_vig_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Vigorous_Dom": round(d_vig_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            }

        activity_minutes["MVPA_ND"] = activity_minutes["Moderate_ND"] + activity_minutes["Vigorous_ND"]
        activity_minutes["MVPA_Dom"] = activity_minutes["Moderate_Dom"] + activity_minutes["Vigorous_Dom"]

        self.activity_volume = activity_minutes
        nd = [activity_minutes["Sedentary_ND"], activity_minutes["Light_ND"],
              activity_minutes["Moderate_ND"], activity_minutes["Vigorous_ND"], activity_minutes["MVPA_ND"]]
        d = [activity_minutes["Sedentary_Dom"], activity_minutes["Light_Dom"],
             activity_minutes["Moderate_Dom"], activity_minutes["Vigorous_Dom"], activity_minutes["MVPA_Dom"]]

        self.activity_df = pd.DataFrame(list(zip(nd, d)), columns=["NonDominant", "Dominant"])
        self.activity_df.index = ["Sedentary", "Light", "Moderate", "Vigorous", "MVPA"]

        self.activity_df["NonDominant%"] = 100 * self.activity_df["NonDominant"] / \
                                           sum(self.activity_df["NonDominant"].loc[["Sedentary", "Light",
                                                                                    "Moderate", "Vigorous"]])
        self.activity_df["Dominant%"] = 100 * self.activity_df["Dominant"] / \
                                        sum(self.activity_df["Dominant"].loc[["Sedentary", "Light",
                                                                              "Moderate", "Vigorous"]])

        self.activity_df["NonDominant%"] = self.activity_df["NonDominant%"].round(2)
        self.activity_df["Dominant%"] = self.activity_df["Dominant%"].round(2)

        print("\nActivity volume:")
        print(self.activity_df)

    def plot_data(self, epoch_len=5, start=None, stop=None, highlight_activity=True, save_plot=True):
        """Plots activity counts from both accelerometers. Saves plot.

           :argument
           -start/stop: timestamp in format YYYY-mm-dd HH:MM:SS
           -epoch_len: epoch length in seconds
           -highlight_activity: boolean for whether to shade regions of non-sedentary activity
           -save_plot: boolean whether to save plot
        """

        # Data cropping and making sure epoch length is correct -------------------------------------------------------
        if epoch_len != self.cutpoint_dict["Epoch length"]:
            self.recalculate_epoch_len(epoch_len=epoch_len, print_statement=True)

        print("\nPlotting all {}-second epoch data...".format(self.cutpoint_dict["Epoch length"]))

        if self.epoched_df is None:
            self.epoched_df = pd.DataFrame(list(zip(self.df_lwrist["Timestamp"],
                                                    self.df_lankle["SVM"],
                                                    self.df_rankle["SVM"],
                                                    self.df_lwrist["SVM"],
                                                    self.df_rwrist["SVM"])),
                                           columns=["Timestamp", "LAnkle", "RAnkle", "LWrist", "RWrist"])

        if start is None and stop is None:
            start = self.epoched_df.iloc[0]["Timestamp"]
            stop = self.epoched_df.iloc[-1]["Timestamp"]

        df = self.epoched_df.loc[(self.epoched_df["Timestamp"] > start) &
                                 (self.epoched_df["Timestamp"] < stop)]

        # Plotting ----------------------------------------------------------------------------------------------------

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(self.fig_width, self.fig_height))
        plt.subplots_adjust(hspace=.35)

        plt.suptitle("All Data: {}-second epochs".format(epoch_len))

        # SHADING NON-SEDENTARY REGIONS -------------------------------------------------------------------------------
        if highlight_activity:
            # Left wrist ----------
            lw = [max(df["LWrist"]) if df["LWrist"].iloc[i] >= self.cutpoint_dict["NonDomLight"] else False
                  for i in range(df.shape[0])]

            ax1.fill_between(x=df["Timestamp"], y1=0, y2=lw, color='purple', alpha=.25)

            # Right wrist --------
            rw = [max(df["RWrist"]) if df["RWrist"].iloc[i] >= self.cutpoint_dict["DomLight"] else False
                  for i in range(df.shape[0])]

            ax2.fill_between(x=df["Timestamp"], y1=0, y2=rw, color='purple', alpha=.25)

        # LEFT WRIST --------------------------------------------------------------------------------------------------
        ax1.set_title("Left Wrist")
        ax1.plot(df["Timestamp"], df["LWrist"], color='black')
        ax1.axhline(y=self.cutpoint_dict["NonDomLight"], color='green', linestyle='dashed')
        ax1.axhline(y=self.cutpoint_dict["NonDomModerate"], color='orange', linestyle='dashed')
        ax1.axhline(y=self.cutpoint_dict["NonDomVigorous"], color='red', linestyle='dashed')
        ax1.set_ylabel("Counts per {} sec".format(epoch_len))

        # RIGHT WRIST -------------------------------------------------------------------------------------------------
        ax2.set_title("Right Wrist")
        ax2.plot(df["Timestamp"], df["RWrist"], color='black')
        ax2.set_ylabel("Counts per {} sec".format(epoch_len))
        ax2.axhline(y=self.cutpoint_dict["DomLight"], color='green', linestyle='dashed')
        ax2.axhline(y=self.cutpoint_dict["DomModerate"], color='orange', linestyle='dashed')
        ax2.axhline(y=self.cutpoint_dict["DomVigorous"], color='red', linestyle='dashed')

        # LEFT ANKLE --------------------------------------------------------------------------------------------------
        ax3.set_title("Left Ankle")
        ax3.plot(df["Timestamp"], df["LAnkle"], color='black')
        ax3.set_ylabel("Counts per {} sec".format(epoch_len))

        # RIGHT WRIST -------------------------------------------------------------------------------------------------
        ax4.set_title("Right Ankle")
        ax4.plot(df["Timestamp"], df["RAnkle"], color='black')
        ax4.set_ylabel("Counts per {} sec".format(epoch_len))

        xfmt = mdates.DateFormatter("%Y/%m/%d \n%H:%M:%S %p")
        ax4.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        if save_plot:
            start_format = datetime.strftime(datetime.strptime(start, "%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H_%M_%S")
            stop_format = datetime.strftime(datetime.strptime(stop, "%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H_%M_%S")

            if highlight_activity:
                f_name = self.check_file_overwrite("AllData_Epoch{}ActivityHighlighted_{}to{}".format(epoch_len,
                                                                                                      start_format,
                                                                                                      stop_format))
            if not highlight_activity:
                f_name = self.check_file_overwrite("AllData_Epoch{}_{}to{}".
                                                   format(epoch_len, start_format, stop_format))

            plt.savefig(f_name)
            print("Plot saved as png ({})".format(f_name))

    def plot_activity_volumes(self, epoch_len=5, plot_type="bar", use_percent=False):
        """Plots activity volumes between non-dominant and dominant wrists.

           :arguments
           -epoch_len: epoch length in seconds
           -plot_type: "bar" or "pie"
           -use_percent: boolean whether to use percent (True) or total minutes (False)
        """

        if epoch_len != self.cutpoint_dict["Epoch length"]:
            self.recalculate_epoch_len(epoch_len=epoch_len, print_statement=True)

        if use_percent:
            suffix = "%"
            units = "%"
        if not use_percent:
            suffix = ""
            units = ' mins'

        if plot_type == "pie":
            plt.subplots(1, 2, figsize=(self.fig_width, self.fig_height))
            plt.suptitle("Activity Volume Comparison: {}-second epochs".format(epoch_len))

            plt.subplot(1, 2, 1)
            plt.title("Non-Dominant Wrist (Left)")
            labels = ["Sed. ({}{})".format(self.activity_df["NonDominant{}".format(suffix)].loc["Sedentary"], units),
                      "Light ({}{})".format(self.activity_df["NonDominant{}".format(suffix)].loc["Light"], units),
                      "Mod. ({}{})".format(self.activity_df["NonDominant{}".format(suffix)].loc["Moderate"], units),
                      "Vig. ({}{})".format(self.activity_df["NonDominant{}".format(suffix)].loc["Vigorous"], units)]
            sizes = [i for i in self.activity_df["NonDominant{}".format(suffix)].iloc[0:4]]
            colors = ['grey', 'green', 'orange', 'red']

            patches, texts = plt.pie(sizes, colors=colors, shadow=False, startangle=90)
            plt.legend(patches, labels, loc="lower left")

            plt.subplot(1, 2, 2)
            plt.title("Dominant Wrist (Right)")
            labels = ["Sed. ({}{})".format(self.activity_df["Dominant{}".format(suffix)].loc["Sedentary"], units),
                      "Light ({}{})".format(self.activity_df["Dominant{}".format(suffix)].loc["Light"], units),
                      "Mod. ({}{})".format(self.activity_df["Dominant{}".format(suffix)].loc["Moderate"], units),
                      "Vig. ({}{})".format(self.activity_df["Dominant{}".format(suffix)].loc["Vigorous"], units)]
            sizes = [i for i in self.activity_df["NonDominant{}".format(suffix)].iloc[0:4]]
            colors = ['grey', 'green', 'orange', 'red']

            patches, texts = plt.pie(sizes, colors=colors, shadow=False, startangle=90)
            plt.legend(patches, labels, loc="lower right")

            plt.show()

        if plot_type == "bar":

            plt.subplots(2, 2, figsize=(self.fig_width, self.fig_height))
            plt.suptitle("Activity Volume Comparison: {}-second epochs".format(epoch_len))

            # Sedentary ----------------------------------------------------------------------------------------------
            plt.subplot(2, 2, 1)
            plt.title("Sedentary")

            plt.bar(["NonDominant", "Dominant"],
                    [self.activity_df.loc["Sedentary"]["NonDominant{}".format(suffix)],
                     self.activity_df.loc["Sedentary"]["Dominant{}".format(suffix)]], edgecolor='black',
                    color=["dimgrey", 'lightgray'])
            plt.ylabel(units)

            plt.subplot(2, 2, 2)
            plt.title("Light")

            plt.bar(["NonDominant", "Dominant"],
                    [self.activity_df.loc["Light"]["NonDominant{}".format(suffix)],
                     self.activity_df.loc["Light"]["Dominant{}".format(suffix)]], edgecolor='black',
                    color=["darkgreen", 'lightgreen'])

            plt.subplot(2, 2, 3)
            plt.title("Moderate")

            plt.bar(["NonDominant", "Dominant"],
                    [self.activity_df.loc["Moderate"]["NonDominant{}".format(suffix)],
                     self.activity_df.loc["Moderate"]["Dominant{}".format(suffix)]], edgecolor='black',
                    color=["darkorange", 'moccasin'])
            plt.ylabel(units)

            plt.subplot(2, 2, 4)
            plt.title("Vigorous")

            plt.bar(["NonDominant", "Dominant"],
                    [self.activity_df.loc["Vigorous"]["NonDominant{}".format(suffix)],
                     self.activity_df.loc["Vigorous"]["Dominant{}".format(suffix)]], edgecolor='black',
                    color=["darkred", 'lightcoral'])


os.chdir("/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 7/")

x = Data(leftankle_filepath="AI_LAnkle_Epoch1s.csv",
         rightankle_filepath="AI_RAnkle_Epoch1s.csv",
         leftwrist_filepath="AI_LWrist_Epoch1s.csv",
         rightwrist_filepath="AI_RWrist_Epoch1s.csv")

# Able to pick any epoch length (seconds). To save as csv: write_file=True
# x.recalculate_epoch_len(epoch_len=5, write_file=False)

# x.plot_data(epoch_len=5, start="2020-10-05 13:05:01", stop="2020-10-05 14:00:00",
#             highlight_activity=False, save_plot=True)

# Plots activity volumes as pie or bar graph from last specified region of data
# x.plot_activity_volumes(epoch_len=15, plot_type="pie", use_percent=True)
