from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statistics
import scipy.stats
from datetime import datetime
import matplotlib.dates as mdates
import pyedflib
from datetime import timedelta
import os
import sys
import warnings
warnings.filterwarnings('ignore')


class Subject:

    def __init__(self, event_filename=None, processed_filename=None, age=71, fig_width=10, fig_height=7):

        self.filename = processed_filename
        self.event_filename = event_filename
        self.fig_width = fig_width
        self.fig_height = fig_height

        self.df_epoch = None
        self.df_event = None
        self.age = 71
        self.rest_hr = 50

        self.hr_zones = {"Light": 0, "Moderate": 0, "Vigorous": 0}

        # Scaled to 75 Hz; default 15-second epochs
        self.cutpoint_dict = {"NonDomLight": round(47 * 75 / 30, 2),
                              "NonDomModerate": round(64 * 75 / 30, 2),
                              "NonDomVigorous": round(157 * 75 / 30, 2)}

        self.activity_volume = None

        # Runs methods -----------------------------------------------------------------------------------------------
        self.import_processed_data()

        if "Sleep_Status" not in self.df_epoch.keys() and self.event_filename is not None:
            self.flag_sleep()
        if "Wear_Status" not in self.df_epoch.keys() and self.event_filename is not None:
            self.flag_nonwear()

        self.calculate_resting_hr()

        self.calculate_hrr()

    def import_processed_data(self):

        print("\nImporting data from processed...")

        self.df_epoch = pd.read_csv(self.filename)
        self.df_epoch["Timestamp"] = [pd.to_datetime(self.df_epoch["Timestamp"].iloc[0]) + timedelta(seconds=15*i) for
                                      i in range(0, self.df_epoch.shape[0])]

        if self.event_filename is not None:
            self.df_event = pd.read_csv(self.event_filename)

            self.df_event["Start"] = pd.to_datetime(self.df_event["Start"])
            self.df_event["Stop"] = pd.to_datetime(self.df_event["Stop"])

        print("Complete.")

    def flag_nonwear(self):

        print("\nFlagging non-wear periods...")

        if self.df_event is not None:
            df_nonwear = self.df_event.loc[self.df_event["Event"] == "Nonwear"]

            df_nonwear["Start"] = pd.to_datetime(df_nonwear["Start"])
            df_nonwear["Stop"] = pd.to_datetime(df_nonwear["Stop"])

            self.df_epoch["Wear_Status"] = np.zeros(self.df_epoch.shape[0])

            for row in df_nonwear.itertuples():
                df = self.df_epoch.loc[(self.df_epoch["Timestamp"] >= row.Start) &
                                       (self.df_epoch["Timestamp"] <= row.Stop)]

                self.df_epoch["Wear_Status"].iloc[df.index[0]:df.index[-1]] = ["Nonwear" for i in
                                                                               range(df.index[-1] - df.index[0])]

            self.df_epoch["Wear_Status"] = ["Wear" if i == 0 else "Nonwear" for i in self.df_epoch["Wear_Status"]]

        print("Complete.")

    def flag_sleep(self):

        print("\nFlagging sleep periods...")

        if self.df_event is not None:
            df_sleep = self.df_event.loc[self.df_event["Event"] == "Sleep"]

            df_sleep["Start"] = pd.to_datetime(df_sleep["Start"])
            df_sleep["Stop"] = pd.to_datetime(df_sleep["Stop"])

            self.df_epoch["Sleep_Status"] = np.zeros(self.df_epoch.shape[0])

            for row in df_sleep.itertuples():
                df = self.df_epoch.loc[(self.df_epoch["Timestamp"] >= row.Start) &
                                       (self.df_epoch["Timestamp"] <= row.Stop)]

                self.df_epoch["Sleep_Status"].iloc[df.index[0]:df.index[-1]] = ["Sleep" for i in
                                                                               range(df.index[-1] - df.index[0])]

            self.df_epoch["Sleep_Status"] = ["Awake" if i == 0 else "Sleep" for i in self.df_epoch["Sleep_Status"]]

        print("Complete.")

    def calculate_resting_hr(self):

        hrs = [i for i in self.df_epoch["HR"]]

        roll_avg = [sum(hrs[i:i+4])/4 if True not in np.isnan(hrs[i:i+4]) else None for i in range(len(hrs)-3)]

        sorted_hrs = sorted([i for i in roll_avg if i is not None])

        low30 = sorted_hrs[0:30]

        self.rest_hr = round(sum(low30)/30, 1)

    def calculate_hrr(self):

        hrr = 208 - .7 * self.age - self.rest_hr

        # HR zones for 30, 40, 60% HRR
        self.hr_zones["Light"] = self.rest_hr + .3 * hrr
        self.hr_zones["Moderate"] = self.rest_hr + .4 * hrr
        self.hr_zones["Vigorous"] = self.rest_hr + .6 * hrr

        # Calculates %HRR from HR
        hrr_list = np.array([round(100 * (hr - self.rest_hr) / hrr, 1) if not np.isnan(hr) else None for
                             hr in self.df_epoch["HR"]])

        for i, hr in enumerate(hrr_list):
            if hr is not None and hr < 0:
                hrr_list[i] = 0

        self.df_epoch["HRR"] = hrr_list

        # Calculates HR intensity
        hr_int = []
        for hr in self.df_epoch["HR"]:
            if np.isnan(hr):
                hr_int.append(None)
            if hr is not None:
                if hr < self.hr_zones["Light"]:
                    hr_int.append(0)
                if self.hr_zones["Light"] <= hr < self.hr_zones["Moderate"]:
                    hr_int.append(1)
                if self.hr_zones["Moderate"] <= hr < self.hr_zones["Vigorous"]:
                    hr_int.append(2)
                if self.hr_zones["Vigorous"] <= hr:
                    hr_int.append(3)

        self.df_epoch["HR_Intensity"] = hr_int

    def plot_time_series(self, start=None, stop=None):

        if start is not None and stop is not None:
            df = self.df_epoch.loc[(self.df_epoch["Timestamp"] >= start) & (self.df_epoch["Timestamp"] <= stop)]
            start = pd.to_datetime(start)
            stop = pd.to_datetime(stop)

        if start is None and stop is None:
            start = self.df_epoch["Timestamp"].iloc[0]
            stop = self.df_epoch["Timestamp"].iloc[-1]
            df = self.df_epoch

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(self.fig_width, self.fig_height))
        plt.suptitle("Time Series LWrist and HR data (sleep = blue; LWrist nonwear = grey)")
        ax1.plot(df["Timestamp"], df['LWrist'], color='black')
        ax1.set_ylabel("Counts")
        ax1.set_title("LWrist")

        ax2.plot(df["Timestamp"], df["HR"], color='red')
        ax2.set_ylabel("HR (bpm)")
        ax2.set_title("Heart Rate")

        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
        ax2.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        for row in self.df_event.itertuples():

            if row.Event == "Sleep" and (start <= row.Start <= stop or start <= row.Stop <= stop):
                ax1.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, max(self.df_epoch["LWrist"])],
                                  color='navy', alpha=.5)
                ax2.fill_betweenx(x1=row.Start, x2=row.Stop, y=[40, max(self.df_epoch["HR"].dropna()) * 1.1],
                                  color='navy', alpha=.5)
            if row.Event == "Nonwear" and (start <= row.Start <= stop or start <= row.Stop <= stop):
                ax1.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, max(self.df_epoch["LWrist"])],
                                  color='grey', alpha=.5)
                ax2.fill_betweenx(x1=row.Start, x2=row.Stop, y=[40, max(self.df_epoch["HR"].dropna()) * 1.1],
                                  color='grey', alpha=.5)

        window_len = (stop - start).total_seconds()
        ax2.set_xlim(start + timedelta(seconds=-window_len/20), stop + timedelta(seconds=window_len/20))

    def compare_wrist_hr_intensity(self, start=None, stop=None, remove_invalid_hr=False):

        if start is not None and stop is not None:
            df = self.df_epoch.loc[(self.df_epoch["Timestamp"] >= start) & (self.df_epoch["Timestamp"] <= stop)].copy()
            start = pd.to_datetime(start)
            stop = pd.to_datetime(stop)

        if start is None and stop is None:
            start = self.df_epoch["Timestamp"].iloc[0]
            stop = self.df_epoch["Timestamp"].iloc[-1]
            df = self.df_epoch.copy()

        if remove_invalid_hr:
            edited_wrist = []
            for row in df.itertuples():
                if row.ECG_Validity == "Valid":
                    edited_wrist.append(row.Wrist_Intensity)
                if row.ECG_Validity == "Invalid":
                    edited_wrist.append(None)
            df["Wrist_Intensity"] = edited_wrist

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(self.fig_width, self.fig_height))
        plt.suptitle("Time Series LWrist and HR Intensity Data (invalid ECG removed = {})"
                     "\n(sleep = blue; LWrist nonwear = grey)".format(remove_invalid_hr))
        ax1.plot(df["Timestamp"], df['Wrist_Intensity'], color='black')
        ax1.set_ylabel("Intensity")
        ax1.set_title("LWrist")

        ax2.plot(df["Timestamp"], df["HR_Intensity"], color='red')
        ax2.set_ylabel("Intensity")
        ax2.set_title("Heart Rate")

        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
        ax2.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        for row in self.df_event.itertuples():

            if row.Event == "Sleep" and (start <= row.Start <= stop or start <= row.Stop <= stop):
                ax1.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, 3],
                                  color='navy', alpha=.5)
                ax2.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, 3],
                                  color='navy', alpha=.5)
            if row.Event == "Nonwear" and (start <= row.Start <= stop or start <= row.Stop <= stop):
                ax1.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, 3],
                                  color='grey', alpha=.5)
                ax2.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, 3],
                                  color='grey', alpha=.5)

        ax1.set_yticks(np.arange(0, 4, 1))
        ax2.set_yticks(np.arange(0, 4, 1))

        window_len = (stop - start).total_seconds()
        ax2.set_xlim(start + timedelta(seconds=-window_len/20), stop + timedelta(seconds=window_len/20))

    def calculate_activity_pattern(self, sort_by='hour', show_plot=True, save_csv=False):
        """Calculates average counts and HR by either hour of day or by day of week. Generates histogram."""

        print("\nAnalyzing activity trends by {}...".format(sort_by))

        def sortby_hour():
            accel_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0,
                           13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
            accel_tally = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0,
                          13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}

            hr_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0,
                          13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
            hr_tally = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0,
                       13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}

            for row in self.df_epoch.itertuples():
                if row.Wear_Status == "Wear":
                    accel_dict[row.Timestamp.hour] += row.LWrist
                    accel_tally[row.Timestamp.hour] += 1

                    if row.ECG_Validity == "Valid":
                        hr_dict[row.Timestamp.hour] += row.HR
                        hr_tally[row.Timestamp.hour] += 1

            accel_averages = [s / t if t != 0 else 0 for s, t in zip(accel_dict.values(), accel_tally.values())]

            hr_averages = [s / t if t != 0 else 0 for s, t in zip(hr_dict.values(), hr_tally.values())]

            r = scipy.stats.pearsonr(accel_averages, hr_averages)
            print("-Hourly correlation between average counts and HR: r = {}".format(round(r[0], 3)))

            if show_plot:
                fig, (ax1, ax2) = plt.subplots(2, figsize=(self.fig_width, self.fig_height))
                plt.subplots_adjust(hspace=.25)
                ax1.set_title("Average Hourly Activity Counts")
                ax1.bar(accel_dict.keys(), accel_averages, align='center',
                        edgecolor='black', color='grey', width=1, alpha=.75)
                ax1.set_ylabel("Counts")
                ax1.set_xticks(np.arange(0, 24, 1))

                ax2.set_title("Average Hourly Heart Rate (invalid ECG removed)")
                ax2.bar(hr_dict.keys(), hr_averages, align='center',
                        edgecolor='black', color='red', width=1, alpha=.5)
                ax2.axhline(self.rest_hr, color='black', linestyle='dashed', label="Resting HR")
                ax2.set_ylim(self.rest_hr * .75, )
                ax2.set_xticks(np.arange(0, 24, 1))
                ax2.set_ylabel("HR (bpm)")
                ax2.set_xlabel("Hour of Day")
                ax2.legend()

            if save_csv:
                df = pd.DataFrame(list(zip(accel_dict.keys(), accel_averages, hr_averages)),
                                  columns=["Hour", "Avg Counts", "Avg HR"])

                print("\nSaving hourly activity data (Activity_by_hour.csv)")
                df.to_csv("Activity_by_hour.csv", index=False)

        def sortby_day():
            # Day of week: Monday = 0, Sunday = 6
            day_list = ["Monday", "Tus"]
            accel_dict = {"Monday": 0, "Tuesday": 0, "Wednesday": 0, "Thursday": 0,
                          "Friday": 0, "Saturday": 0, "Sunday": 0}
            accel_tally = {"Monday": 0, "Tuesday": 0, "Wednesday": 0, "Thursday": 0,
                          "Friday": 0, "Saturday": 0, "Sunday": 0}

            hr_dict = {"Monday": 0, "Tuesday": 0, "Wednesday": 0, "Thursday": 0,
                          "Friday": 0, "Saturday": 0, "Sunday": 0}
            hr_tally = {"Monday": 0, "Tuesday": 0, "Wednesday": 0, "Thursday": 0,
                          "Friday": 0, "Saturday": 0, "Sunday": 0}

            for row in self.df_epoch.itertuples():
                if row.Wear_Status == "Wear":
                    accel_dict[row.Timestamp.day_name()] += row.LWrist
                    accel_tally[row.Timestamp.day_name()] += 1

                    if row.ECG_Validity == "Valid":
                        hr_dict[row.Timestamp.day_name()] += row.HR
                        hr_tally[row.Timestamp.day_name()] += 1

            accel_averages = [s / t if t != 0 else 0 for s, t in zip(accel_dict.values(), accel_tally.values())]

            hr_averages = [s / t if t != 0 else 0 for s, t in zip(hr_dict.values(), hr_tally.values())]

            r = scipy.stats.pearsonr(accel_averages, hr_averages)
            print("-Daily correlation between average counts and HR: r = {}".format(round(r[0], 3)))

            if show_plot:
                fig, (ax1, ax2) = plt.subplots(2, figsize=(self.fig_width, self.fig_height))
                plt.subplots_adjust(hspace=.25)

                ax1.set_title("Average Daily Activity Counts")
                ax1.bar(accel_dict.keys(), accel_averages, align='center',
                        edgecolor='black', color='grey', width=1, alpha=.75)
                ax1.set_ylabel("Counts")

                ax2.set_title("Average Daily Heart Rate (invalid ECG removed)")
                ax2.bar(hr_dict.keys(), hr_averages, align='center',
                        edgecolor='black', color='red', width=1, alpha=.5)
                ax2.axhline(self.rest_hr, color='black', linestyle='dashed', label="Resting HR")
                ax2.set_ylim(self.rest_hr * .75, )
                ax2.set_ylabel("HR (bpm)")
                ax2.set_xlabel("Hour of Day")
                ax2.legend()

            if save_csv:
                df = pd.DataFrame(list(zip(accel_dict.keys(), accel_averages, hr_averages)),
                                  columns=["Day", "Avg Counts", "Avg HR"])

                print("\nSaving hourly activity data (Activity_by_hour.csv)")
                df.to_csv("Activity_by_day.csv", index=False)

        if sort_by == "hour":
            sortby_hour()
        if sort_by == "day":
            sortby_day()

    def find_mvpa_bouts(self, min_dur=10, breaktime=2):
        """Finds MVPA bouts of duration min_dur allowing for a break of breaktime minutes. Also finds longest MVPA
           bout and prints result.

           Bout function doesn't actually work since there is no detected MVPA bouts with participant and I'm lazy.
        """

        print("\nFinding MVPA activity bouts with minimum duration of {} minutes with a "
              "{}-minute break allowed...".format(min_dur, breaktime))

        # Finds longest MVPA bout (no breaks)
        longest = 0
        current = 0
        for num in [i for i in self.df_epoch["Wrist_Intensity"]]:
            if num >= 2:
                current += 1
            else:
                longest = max(longest, current)
                current = 0

        print("-No {}-minute bouts founds.".format(min_dur))
        print("-Longest MVPA bout was {} minutes.".format(longest * 15 / 60))

    def analyze_hrv(self):
        """Plots histograms of all epoch's RR SD and during sedentary periods only. Shades in data regions with
           data from Shaffer, F. & Ginsberg, P. (2017). An Overview of Heart Rate Variability Metrics and Norms."""

        df = self.df_epoch.copy()

        plt.subplots(1, 2, figsize=(self.fig_width, self.fig_height))

        plt.subplot(1, 2, 1)
        h = plt.hist(df["RR_SD"].dropna(), bins=np.arange(0, 250, 10),
                     weights=100*np.ones(len(df["RR_SD"].dropna())) / len(df["RR_SD"].dropna()),
                     edgecolor='black', color='grey', alpha=.5, cumulative=False)
        plt.ylabel("% of epochs")
        plt.xlabel("RR SD (ms)")
        plt.title("All data")

        # Shaffer & Ginsberg, 2017 interpretation
        plt.fill_betweenx(x1=0, x2=50, y=[0, plt.ylim()[1]], color='red', alpha=.5,
                          label="Unhealthy ({}%)".format(round(sum(h[0][0:5]), 1)))
        plt.fill_betweenx(x1=50, x2=100, y=[0, plt.ylim()[1]], color='orange', alpha=.5,
                          label="Compromised ({}%)".format(round(sum(h[0][5:10]), 1)))
        plt.fill_betweenx(x1=100, x2=250, y=[0, plt.ylim()[1]], color='green', alpha=.5,
                          label="Healthy ({}%)".format(round(sum(h[0][10:]), 1)))
        plt.legend()

        df = self.df_epoch.dropna()
        df = df.loc[df["HR_Intensity"] == 0]

        plt.subplot(1, 2, 2)
        h = plt.hist(df["RR_SD"].dropna(), bins=np.arange(0, 250, 10),
                     weights=100*np.ones(len(df["RR_SD"].dropna())) / len(df["RR_SD"].dropna()),
                     edgecolor='black', color='grey', alpha=.5, cumulative=False)
        plt.xlabel("RR SD (ms)")
        plt.title("Sedentary only")

        # Shaffer & Ginsberg, 2017 interpretation
        plt.fill_betweenx(x1=0, x2=50, y=[0, plt.ylim()[1]], color='red', alpha=.5,
                          label="Unhealthy ({}%)".format(round(sum(h[0][0:5]), 1)))
        plt.fill_betweenx(x1=50, x2=100, y=[0, plt.ylim()[1]], color='orange', alpha=.5,
                          label="Compromised ({}%)".format(round(sum(h[0][5:10]), 1)))
        plt.fill_betweenx(x1=100, x2=250, y=[0, plt.ylim()[1]], color='green', alpha=.5,
                          label="Healthy ({}%)".format(round(sum(h[0][10:]), 1)))
        plt.legend()

    def analyze_sleep(self):
        """Barplot of nightly sleep durations. Recommended range marked."""

        df = self.df_event.loc[self.df_event["Event"] == "Sleep"]
        df["Durations"] = [(df["Stop"].iloc[i] - df["Start"].iloc[i]).total_seconds() / 3600 for
                           i in range(df.shape[0])]

        plt.bar([df["Start"].iloc[i].day_name() for i in range(df.shape[0])],
                df["Durations"], color='slategrey', alpha=.75, edgecolor='black')
        plt.ylabel("Hours per night")
        plt.title("Nightly Sleep Duration")

        plt.axhline(y=9, color='green', linestyle='dashed', label="Recommended max.")
        plt.axhline(y=7, color='red', linestyle='dashed', label="Recommended min.")
        plt.legend(loc='lower right')

        for sleep in df.itertuples():
            plt.text(x=sleep.Index-.20, y=4, s=str(round(sleep.Durations, 1)) + "\nhours")

    def calculate_activity_volume(self, remove_invalid_ecg=True, show_plot=True):
        """Calculates activity volumes (minutes and percent of data) for LWrist and HR data. Able to crop.

           :argument
           -remove_invalid_ecg: boolean whether to include invalid ECG signal periods.
                                If True, the total volume of data will be the same between LWrist and HR data.
                                If False, LWrist contains more data.
        """

        df = self.df_epoch.loc[self.df_epoch["Wear_Status"] == "Wear"]

        if remove_invalid_ecg:
            df = df.loc[df["ECG_Validity"] == "Valid"]

        print("\nCalculating activity data from {} to {} "
              "in 15-second epochs...".format(df.iloc[0]["Timestamp"], df.iloc[-1]['Timestamp']))

        # Non-dominant (left) wrist -----------------------------------------------------------------------------------
        nd_sed_epochs = df["LWrist"].loc[(df["LWrist"] < self.cutpoint_dict["NonDomLight"])].shape[0]

        nd_light_epochs = df["LWrist"].loc[(df["LWrist"] >= self.cutpoint_dict["NonDomLight"]) &
                                           (df["LWrist"] < self.cutpoint_dict["NonDomModerate"])].shape[0]

        nd_mod_epochs = df["LWrist"].loc[(df["LWrist"] >= self.cutpoint_dict["NonDomModerate"]) &
                                         (df["LWrist"] < self.cutpoint_dict["NonDomVigorous"])].shape[0]

        nd_vig_epochs = df["LWrist"].loc[(df["LWrist"] >= self.cutpoint_dict["NonDomVigorous"])].shape[0]

        # Heart rate -------------------------------------------------------------------------------------------------
        hr_sed_epochs = df.loc[df["HRR"] < 30].shape[0]
        hr_light_epochs = df.loc[(df["HRR"] >= 30) & (df["HRR"] < 40)].shape[0]
        hr_mod_epochs = df.loc[(df["HRR"] >= 40) & (df["HRR"] < 60)].shape[0]
        hr_vig_epochs = df.loc[df["HRR"] >= 60].shape[0]

        # Data storage ------------------------------------------------------------------------------------------------
        activity_minutes = {"LWristSed": round(nd_sed_epochs / 4, 2),
                            "LWristLight": round(nd_light_epochs / 4, 2),
                            "LWristMod": round(nd_mod_epochs / 4, 2),
                            "LWristVig": round(nd_vig_epochs / 4, 2),
                            "HRSed": round(hr_sed_epochs / 4, 2),
                            "HRLight": round(hr_light_epochs / 4, 2),
                            "HRMod": round(hr_mod_epochs / 4, 2),
                            "HRVig": round(hr_vig_epochs / 4, 2),
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

    def calculate_daily_activity_volume(self, save_csv=False):

        print("\nCalculating daily activity volumes...")

        date_list = sorted([j for j in set(i.date() for i in self.df_epoch["Timestamp"])])

        # Sets df
        df = self.df_epoch.copy()

        df["Date"] = [i.date() for i in self.df_epoch["Timestamp"]]

        # Calculates daily activity
        wrist_sed = []
        wrist_light = []
        wrist_mvpa = []
        hr_sed = []
        hr_light = []
        hr_mvpa = []

        for date in date_list:
            data = df.loc[df["Date"] == date]

            values_wrist = [i/4 for i in pd.value_counts(data["Wrist_Intensity"])]
            values_hr = [i/4 for i in pd.value_counts(data["HR_Intensity"])]

            wrist_sed.append(values_wrist[0])
            wrist_light.append(values_wrist[1])
            wrist_mvpa.append(sum(values_wrist[2:]))

            hr_sed.append(values_hr[0])
            hr_light.append(values_hr[1])
            hr_mvpa.append(sum(values_hr[2:]))

        # PLOTTING ---------------------------------------------------------------------------------------------------
        day_list = [j for j in [datetime.strftime(i, "%a") for i in date_list]]
        plt.subplots(2, 3, figsize=(self.fig_width, self.fig_height))
        plt.subplots_adjust(hspace=.25)
        plt.title("Daily Activity")

        # WRIST
        plt.subplot(2, 3, 1)
        plt.bar(day_list, wrist_sed, color='grey', edgecolor='black')
        plt.title("Wrist Sedentary")
        plt.ylabel("Minutes")

        plt.subplot(2, 3, 2)
        plt.bar(day_list, wrist_light, color='green', edgecolor='black')
        plt.title("Wrist Light")

        plt.subplot(2, 3, 3)
        plt.bar(day_list, wrist_mvpa, color='orange', edgecolor='black')
        plt.title("Wrist MVPA")

        # HR
        plt.subplot(2, 3, 4)
        plt.bar(day_list, hr_sed, color='grey', edgecolor='black')
        plt.title("HR Sedentary")
        plt.ylabel("Minutes")

        plt.subplot(2, 3, 5)
        plt.bar(day_list, hr_light, color='green', edgecolor='black')
        plt.title("HR Light")

        plt.subplot(2, 3, 6)
        plt.bar(day_list, hr_mvpa, color='orange', edgecolor='black')
        plt.title("HR MVPA")

        # Dataframe ---------------------------------------------------------------------------------------------------
        df_vol = pd.DataFrame(list(zip(day_list, wrist_sed, hr_sed, wrist_light, hr_light, wrist_mvpa, hr_mvpa)),
                              columns=["Day", "Wrist_Sed", "HR_Sed", "Wrist_Light",
                                       "HR_Light", "Wrist_MVPA", "HR_MVPA"])

        if save_csv:
            df_vol.to_csv("DailyActivityVolume.csv", index=False)
            print("Saved file as DailyActivityVolume.csv.")


x = Subject(processed_filename="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 9/Lab9_Epoched.csv",
            event_filename="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 9/Lab9_EventLog.csv")

# Plots LWrist counts and HR. Sleep/nonwear periods shaded.
# x.plot_time_series(start=None, stop=None)

# Plots LWrist and HR intensity. Sleep/nonwear periods shaded
# x.compare_wrist_hr_intensity(remove_invalid_hr=False, start="2020-03-03 14:00:00", stop="2020-03-03 15:00:00")

# Analyze activity patterns by 'day' or by 'hour'
# x.calculate_activity_pattern(sort_by='hour', show_plot=True, save_csv=False)

# Finds MVPA bouts of duration min_dur with allowance for break of duration breaktime (minutes)
# x.find_mvpa_bouts(min_dur=10, breaktime=2)

# Calculates total activity volumes for LWrist and HR data
# x.calculate_activity_volume()

# Calculates daily activity volumes for LWrist and HR data. Able to save as csv
# x.calculate_daily_activity_volume(save_csv=False)
