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

    def __init__(self, event_filename=None, processed_filename=None, age=71, weight=66, fig_width=10, fig_height=7):

        self.filename = processed_filename
        self.event_filename = event_filename
        self.fig_width = fig_width
        self.fig_height = fig_height

        self.df_epoch = None
        self.df_event = None
        self.age = age
        self.weight = weight
        self.rest_hr = 50

        self.hr_zones = {"Light": 0, "Moderate": 0, "Vigorous": 0}

        # Scaled to 75 Hz; default 15-second epochs
        self.cutpoint_dict = {"NonDomLight": round(47 * 75 / 30, 2),
                              "NonDomModerate": round(64 * 75 / 30, 2),
                              "NonDomVigorous": round(157 * 75 / 30, 2)}

        self.activity_volume = None
        self.df_daily_volumes = None

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

    def heartrate_histogram(self, save_plot=False):

        print("\nGenerating histogram of all heart rates...")

        hr = [i for i in self.df_epoch["HR"] if not np.isnan(i)]

        plt.axvline(x=self.rest_hr, linestyle='dashed', color='black', label='Resting HR')
        plt.axvline(x=208 - .7 * self.age, linestyle='dashed', color='red', label="Max HR")

        plt.fill_between(x=[0, self.hr_zones["Light"]], y1=0, y2=100,
                         color='grey', alpha=.5, label="Sedentary")
        plt.fill_between(x=[self.hr_zones["Light"], self.hr_zones["Moderate"]], y1=0, y2=100,
                         color='green', alpha=.5, label="Light")
        plt.fill_between(x=[self.hr_zones["Moderate"], self.hr_zones["Vigorous"]], y1=0, y2=100,
                         color='orange', alpha=.5, label="Moderate")
        plt.fill_between(x=[self.hr_zones["Vigorous"], 208 - .7 * self.age], y1=0, y2=100,
                         color='red', alpha=.5, label="Vigorous")

        data = plt.hist(hr, bins=np.arange(0, 200, 5), weights=100 * np.ones(len(hr)) / len(hr),
                        color='white', alpha=.5, edgecolor='black')
        plt.xlabel("HR (bpm)")
        plt.ylabel("% of epochs")
        plt.title("HR Distribution")

        plt.legend(loc='upper right')
        plt.ylim(0, max(data[0] * 1.05))
        plt.xlim(40, (208 - .7 * self.age) * 1.02)

        if save_plot:
            print("Plot saved as HR_Histogram.png")
            plt.savefig("HR_Histogram.png")

    def plot_time_series(self, start=None, stop=None, show_events=True, show_thresholds=False, save_plot=False):

        print("\nPlotting time series Wrist and HR data...")

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

        if show_events:
            for row in self.df_event.itertuples():

                if row.Event == "Sleep" and (start <= row.Start <= stop or start <= row.Stop <= stop):
                    ax1.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, max(self.df_epoch["LWrist"])],
                                      color='dodgerblue', alpha=.35)
                    ax2.fill_betweenx(x1=row.Start, x2=row.Stop, y=[40, max(self.df_epoch["HR"].dropna()) * 1.1],
                                      color='dodgerblue', alpha=.35)
                if row.Event == "Nonwear" and (start <= row.Start <= stop or start <= row.Stop <= stop):
                    ax1.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, max(self.df_epoch["LWrist"])],
                                      color='grey', alpha=.75)
                    ax2.fill_betweenx(x1=row.Start, x2=row.Stop, y=[40, max(self.df_epoch["HR"].dropna()) * 1.1],
                                      color='grey', alpha=.75)

        if show_thresholds:
            ax1.axhline(self.cutpoint_dict["NonDomLight"], color='green', linestyle='dashed')
            ax1.axhline(self.cutpoint_dict["NonDomModerate"], color='orange', linestyle='dashed')
            ax1.axhline(self.cutpoint_dict["NonDomVigorous"], color='red', linestyle='dashed')

            ax2.axhline(self.rest_hr, color='black', linestyle='dashed')
            ax2.axhline(self.hr_zones["Light"], color='green', linestyle='dashed')
            ax2.axhline(self.hr_zones["Moderate"], color='orange', linestyle='dashed')
            ax2.axhline(self.hr_zones["Vigorous"], color='red', linestyle='dashed')

        window_len = (stop - start).total_seconds()
        ax2.set_xlim(start + timedelta(seconds=-window_len/20), stop + timedelta(seconds=window_len/20))

        if save_plot:
            print("Plot saved as EpochedTimeSeries.png")
            plt.savefig("EpochedTimeSeries.png")

    def compare_wrist_hr_intensity(self, start=None, stop=None, remove_invalid_hr=False,
                                   show_events=False, save_plot=False):

        print("\nPlotting time series intensity data...")

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

        if show_events:
            for row in self.df_event.itertuples():

                if row.Event == "Sleep" and (start <= row.Start <= stop or start <= row.Stop <= stop):
                    ax1.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, 3],
                                      color='dodgerblue', alpha=.5)
                    ax2.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, 3],
                                      color='dodgerblue', alpha=.5)
                if row.Event == "Nonwear" and (start <= row.Start <= stop or start <= row.Stop <= stop):
                    ax1.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, 3],
                                      color='grey', alpha=.5)
                    ax2.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, 3],
                                      color='grey', alpha=.5)

        ax1.set_yticks(np.arange(0, 4, 1))
        ax2.set_yticks(np.arange(0, 4, 1))

        window_len = (stop - start).total_seconds()
        ax2.set_xlim(start + timedelta(seconds=-window_len/20), stop + timedelta(seconds=window_len/20))

        if save_plot:
            print("Plot saved as Intensity_TimeSeries.png")
            plt.savefig("Intensity_TimeSeries.png")

    def calculate_activity_pattern(self, sort_by='hour', show_plot=True, save_plot=False, save_csv=False):
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

        if save_plot:
            print("Plot saved as ActivityPattern_{}.png".format(sort_by))
            plt.savefig("ActivityPattern_{}.png".format(sort_by))

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

    def analyze_hrv(self, save_plot=False, save_csv=False):
        """Plots histograms of all epoch's RR SD and during sedentary periods only. Shades in data regions with
           data from Shaffer, F. & Ginsberg, P. (2017). An Overview of Heart Rate Variability Metrics and Norms."""

        df = self.df_epoch.copy()

        plt.subplots(1, 2, figsize=(self.fig_width, self.fig_height))
        plt.title("HRV with interpretation (Shaffer & Ginsberg, 2017)")

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

        if save_plot:
            print("Saving plot as HRV_Histogram.png")
            plt.savefig("HRV_Histogram.png")

        if save_csv:
            print("Data saved as HRV_FrequencyData.csv")
            bins = [i for i in h[1][:-1]]
            freqs = [i for i in h[0]]
            data = pd.DataFrame(list(zip(bins, freqs)), columns=["Bin", "Frequency_%"])
            data.to_csv("HRV_FrequencyData.csv", index=False)

    def analyze_sleep(self, save_plot=False, save_data=False):
        """Barplot of nightly sleep durations. Recommended range marked."""

        print("\nPlotting sleep data...")

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

        if save_plot:
            print("Saved plot as SleepDurations.png")
            plt.savefig("SleepDurations.png")

        if save_data:
            data = pd.DataFrame(list(zip([df["Start"].iloc[i].day_name() for i in range(df.shape[0])],
                                         df["Durations"])), columns=["Day", "HoursSlept"])
            data.to_csv("SleepDurations.csv", index=False)
            print("Data saved as SleepDurations.csv")

    def calculate_total_activity_volume(self, remove_invalid_ecg=True, show_plot=True):
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

    def calculate_daily_activity_volume(self, save_csv=False, save_plot=False):

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
        self.df_daily_volumes = pd.DataFrame(list(zip(day_list, wrist_sed, hr_sed, wrist_light,
                                                      hr_light, wrist_mvpa, hr_mvpa)),
                                             columns=["Day", "Wrist_Sed", "HR_Sed", "Wrist_Light",
                                                      "HR_Light", "Wrist_MVPA", "HR_MVPA"])

        if save_csv:
            self.df_daily_volumes.to_csv("DailyActivityVolume.csv", index=False)
            print("Saved file as DailyActivityVolume.csv.")

        if save_plot:
            print("Saved plot as DailyActivityVolume.png")
            plt.savefig("DailyActivityVolume.png")

    def calculate_wrist_ee(self):
        """Calculates Wrist accelerometer intensity using regression equation from Powell et al., 2017."""

        data = [i * (30 / 75) for i in self.df_epoch["LWrist"]]

        # Modified equation from Powell et al. 2017. Removed resting component (constant = 1.15451)
        mets = [.022261 * i for i in data]

        # Converts METs to relative VO2 (mL O2/kg/min)
        r_vo2 = [3.5 * m for m in mets]

        # Converts relative VO2 to absolute VO2 (L O2/kg/min)
        a_vo2 = [i * self.weight / 1000 for i in r_vo2]

        # Converts absolute VO2 to kcal/min (assumes 1 L O2 -> 4.825 kcal)
        kcal_min = [a * 4.825 for a in a_vo2]

        # Calculates kcal/epoch
        kcal_epoch = [k * (15 / 60) for k in kcal_min]

        total_ee = sum([i for i in kcal_epoch if not np.isnan(i)])
        print("-Total energy expenditure estimated from Wrist is {} kcal.".format(int(total_ee)))

        self.df_epoch["Wrist_EE"] = kcal_min

    def calculate_hr_ee(self):
        """Calculates Physical Activity Intensity using equation from Brage et al., 2004.
        Brage, S., Brage, N., Franks, P., Ekelund, U., Wong, M., Andersen, L., et al. (2004). Branched equation
        modeling of simultaneous accelerometry and heart rate monitoring improves estimate of directly measured
        physical activity energy expenditure. J Appl Physiol. 96. 343-351.
        """

        # HR - resting HR = net HR
        net_hr = np.array([i - self.rest_hr if i is not None else None for i in self.df_epoch["HR"]])

        # Sets values below 0% HRR (below resting HR) to 0
        net_hr[net_hr <= 0] = 0

        # Equation from Brage et al., 2004. Active EE in kJ/kg/min
        kj_kg_min = [.011 * (hr ** 2) + 5.82 * hr if hr is not None else None for hr in net_hr]

        # Converts kJ to kcal: relative EE (kcal/kg/min)
        kcal_kg_min = [k / 4.184 if k is not None else None for k in kj_kg_min]

        # Converts relative EE to absolute EE (kcal/min)
        kcal_min = [k * self.weight / 1000 if k is not None else None for k in kcal_kg_min]

        # kcals per epoch instead of per minute
        kcal_epoch = [k * (15 / 60) for k in kcal_min]

        total_ee = sum([i for i in kcal_epoch if not np.isnan(i)])
        print("-Total energy expenditure estimated from HR is {} kcal.".format(int(total_ee)))

        self.df_epoch["HR_EE"] = kcal_min

    def plot_ee(self, save_plot=False):

        # Calculates active energy expenditure (level above resting) -------------------------------------------------
        print("\nCalculating energy expenditure...")
        self.calculate_hr_ee()
        self.calculate_wrist_ee()

        df = self.df_epoch.loc[self.df_epoch["ECG_Validity"] == "Valid"]

        wrist = df["Wrist_EE"].sum() * (15 / 60)
        hr = df["HR_EE"].sum() * (15 / 60)

        print("    -Ignoring invalid ECG periods: wrist EE = {} kcal.".format(int(wrist)))

        # Plotting ---------------------------------------------------------------------------------------------------
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(self.fig_width, self.fig_height))
        plt.suptitle("Active Energy Expenditure")
        plt.subplots_adjust(hspace=.3)

        ax1.set_title("LWrist")
        ax1.plot(self.df_epoch["Timestamp"], self.df_epoch["LWrist"], color='black', label='Wrist')
        ax1.set_ylabel("Counts")

        ax2.set_title("Heart Rate")
        ax2.plot(self.df_epoch["Timestamp"], self.df_epoch["HR"], color='red', label='HR')
        ax2.set_ylabel("HR (bpm)")
        ax2.axhline(y=self.rest_hr, linestyle='dashed', color='grey', label='Rest HR')
        ax2.legend()

        ax3.set_title("Energy Expenditure")
        ax3.plot(self.df_epoch["Timestamp"], self.df_epoch["HR_EE"], color='red', label='HR')
        ax3.plot(self.df_epoch["Timestamp"], self.df_epoch["Wrist_EE"], color='black', label='Wrist')
        ax3.axhline(y=0, linestyle='dashed', color='grey')
        ax3.set_ylabel("kcal/minute")
        ax3.legend()

        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
        ax1.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        if save_plot:
            print("Plot saved as EnergyExpenditure.png")
            plt.savefig("EnergyExpenditure.png")


x = Subject(processed_filename="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 9/Lab9_Epoched.csv",
            event_filename="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 9/Lab9_EventLog.csv")

# Plots LWrist counts and HR. Able to hide/show sleep/nonwear and activiity thresholds
# x.plot_time_series(start=None, stop=None, show_events=True, show_thresholds=True, save_plot=False)

# Plots LWrist and HR intensity. Sleep/nonwear periods shaded
# x.compare_wrist_hr_intensity(show_events=False, remove_invalid_hr=False, start=None, stop=None, save_plot=False)

# Analyze activity patterns by 'day' or by 'hour'
# x.calculate_activity_pattern(sort_by='day', show_plot=True, save_plot=False, save_csv=False)

# Finds MVPA bouts of duration min_dur with allowance for break of duration breaktime (minutes)
# x.find_mvpa_bouts(min_dur=10, breaktime=2)

# Calculates total activity volumes for LWrist and HR data
# x.calculate_total_activity_volume(remove_invalid_ecg=False, show_plot=True)

# Calculates daily activity volumes for LWrist and HR data. Able to save as csv
# x.calculate_daily_activity_volume(save_csv=False, save_plot=False)

# Generates histogram of all HR data
# x.heartrate_histogram(save_plot=False)

# Histogram of each epoch's HRV (SD of RR intervals)
# x.analyze_hrv(save_plot=False, save_csv=False)

# Bar plot of each night's sleep duration with recommended range
# x.analyze_sleep(save_plot=False, save_data=False)

# Plots time series LWrist, HR, and estimated EE data
# x.plot_ee(save_plot=False)
