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

        self.start_stamp = None
        self.stop_stamp = None

        self.epoched_df = None  # dataframe of all devices for one epoch length

        self.activity_volume = None  # activity volume for one epoch length

        self.df_all_volumes = None  # activity volumes for 1, 5, 15, 30, and 60-second epochs

        # Methods and objects that are run automatically when class instance is created -------------------------------

        self.df_lankle, self.lankle_samplerate = self.load_correct_file(filepath=self.lankle_fname,
                                                                        f_type="Left Ankle")
        self.df_lwrist, self.lwrist_samplerate = self.load_correct_file(filepath=self.lwrist_fname,
                                                                        f_type="Left Wrist")
        self.df_rankle, self.rankle_samplerate = self.load_correct_file(filepath=self.rankle_fname,
                                                                        f_type="Right Ankle")
        self.df_rwrist, self.rwrist_samplerate = self.load_correct_file(filepath=self.rwrist_fname,
                                                                        f_type="Right Wrist")

        # Crops dataframes so length is a multiple of 1, 5, 10, 15, 30, and 60-second epochs
        # self.crop_dataframes()

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
                                       columns=["Timestamp", "LAnkle", "Rankle", "LWrist", "RWrist"])

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

    def scale_cutpoints(self):
        pass

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

        activity_minutes = {"Sedentary_ND": round(nd_sed_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Sedentary_Dom": round(d_sed_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Light_ND": round(nd_light_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Light_Dom": round(d_light_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Moderate_ND": round(nd_mod_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Moderate_Dom": round(d_mod_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Vigorous_ND": round(nd_vig_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2),
                            "Vigorous_Dom": round(d_vig_epochs / (60 / self.cutpoint_dict["Epoch length"]), 2)
                            }

        self.activity_volume = activity_minutes
        nd = [activity_minutes["Sedentary_ND"], activity_minutes["Light_ND"],
              activity_minutes["Moderate_ND"], activity_minutes["Vigorous_ND"]]
        d = [activity_minutes["Sedentary_Dom"], activity_minutes["Light_Dom"],
             activity_minutes["Moderate_Dom"], activity_minutes["Vigorous_Dom"]]
        activity_df = pd.DataFrame(list(zip(nd, d)), columns=["NonDominant", "Dominant"])
        activity_df.index = ["Sedentary", "Light", "Moderate", "Vigorous"]

        print("\nActivity volume:")
        print(activity_df)

    def plot_data(self):
        pass


os.chdir("/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 7/")

x = Data(leftankle_filepath="AI_LAnkle_Epoch1s.csv",
         rightankle_filepath="AI_RAnkle_Epoch1s.csv",
         leftwrist_filepath="AI_LWrist_Epoch1s.csv",
         rightwrist_filepath="AI_RWrist_Epoch1s.csv")

# Able to pick any epoch length (seconds). To save as csv: write_file=True
x.recalculate_epoch_len(epoch_len=5, write_file=False)
