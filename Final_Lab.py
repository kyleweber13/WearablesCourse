from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statistics
import scipy.stats as stats
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

        # Runs methods -----------------------------------------------------------------------------------------------
        self.import_processed_data()

        if "Sleep_Status" not in self.df_epoch.keys() and self.event_filename is not None:
            self.flag_sleep()
        if "Wear_Status" not in self.df_epoch.keys() and self.event_filename is not None:
            self.flag_nonwear()

        self.calculate_resting_hr()

        self.calculate_hrr()

    def import_processed_data(self):

        self.df_epoch = pd.read_csv(self.filename)
        self.df_epoch["Timestamp"] = pd.to_datetime(self.df_epoch["Timestamp"])

        if self.event_filename is not None:
            self.df_event = pd.read_csv(self.event_filename)

            self.df_event["Start"] = pd.to_datetime(self.df_event["Start"])
            self.df_event["Stop"] = pd.to_datetime(self.df_event["Stop"])

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
                ax2.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, max(self.df_epoch["LWrist"])],
                                  color='navy', alpha=.5)
            if row.Event == "Nonwear" and (start <= row.Start <= stop or start <= row.Stop <= stop):
                ax1.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, max(self.df_epoch["LWrist"])],
                                  color='grey', alpha=.5)
                ax2.fill_betweenx(x1=row.Start, x2=row.Stop, y=[0, max(self.df_epoch["LWrist"])],
                                  color='grey', alpha=.5)

        window_len = (stop - start).total_seconds()
        ax2.set_xlim(start + timedelta(seconds=-window_len/20), stop + timedelta(seconds=window_len/20))


x = Subject(processed_filename="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 9/Lab9_Epoched.csv",
            event_filename="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab 9/Lab9_EventLog.csv")

# Plots LWrist counts and HR. Sleep/nonwear periods shaded.
# x.plot_time_series(start=None, stop=None)
