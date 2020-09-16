import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
from scipy.signal import butter, filtfilt


class FilterTest:

    def __init__(self, freqs=(), filter_type="lowpass", low_f=1, high_f=10, filter_order=3):
        """Class that is designed to demonstrate how sine wave addition, filtering, and FFTs work.
            Plots raw data, filtered data, raw data FFT, and filtered data FFT on single plot.

        :arguments
        -freqs: list of frequencies to include in the raw data. e.g. freqs=[1, 3, 5]
        -filter_type: type of filter; "lowpass", "highpass", or "bandpass"
        -low_f: frequency used if filter_type=="lowpass" or low-end cutoff frequency for "bandpass"
        -high_f: frequency used if filter_type=="highpass" or high-end cutoff frequency for "bandpass"
        -filter_order: integer or float
        """

        self.length = 375  # Number of samples
        self.x = np.arange(self.length)
        self.fs = 75  # sampling rate

        self.freqs = freqs
        self.filter_type = filter_type
        self.low_f = low_f
        self.high_f = high_f
        self.filter_order = filter_order

        self.raw_data = []
        self.filtered_data = []

        self.raw_fft = None
        self.filtered_fft = None

        self.data_freq = []
        self.filter_details = None

        # RUNS METHODS
        self.create_raw_wave()
        self.filter_signal()
        self.plot_data()

    def create_raw_wave(self):
        """Creates a sine wave that is the addition of all frequencies specified in self.freqs"""

        print("Generating signal from sine wave(s) with {} Hz frequency.".format(self.freqs))
        data = []

        for f in self.freqs:
            y = np.sin(2 * np.pi * f * self.x / self.fs)
            data.append(y)

        net_wave = np.sum(data, axis=0)

        self.raw_data = net_wave
        self.data_freq = self.freqs

        self.raw_fft = scipy.fft.fft(self.raw_data)

    def filter_signal(self):
        """Filters data using details specified by self.filter_type, self.low_f, self.high_f, and self.filter_order.
        """

        nyquist_freq = 0.5 * self.fs

        if self.filter_type == "lowpass":
            print("Running a {}Hz lowpass filter.".format(self.low_f))
            self.filter_details = "{}Hz lowpass".format(self.low_f)

            low = self.low_f / nyquist_freq
            b, a = butter(N=self.filter_order, Wn=low, btype="lowpass")
            # filtered_data = lfilter(b, a, data)
            filtered_data = filtfilt(b, a, x=self.raw_data)

        if self.filter_type == "highpass":
            print("Running a {}Hz highpass filter.".format(self.high_f))
            self.filter_details = "{}Hz highpass".format(self.high_f)

            high = self.high_f / nyquist_freq
            b, a = butter(N=self.filter_order, Wn=high, btype="highpass")
            # filtered_data = lfilter(b, a, data)
            filtered_data = filtfilt(b, a, x=self.raw_data)

        if self.filter_type == "bandpass":
            print("Running a {}-{}Hz bandpass filter.".format(self.low_f, self.high_f))
            self.filter_details = "{}-{}Hz bandpass".format(self.low_f, self.high_f)

            low = self.low_f / nyquist_freq
            high = self.high_f / nyquist_freq

            b, a = butter(N=self.filter_order, Wn=[low, high], btype="bandpass")
            # filtered_data = lfilter(b, a, data)
            filtered_data = filtfilt(b, a, x=self.raw_data)

        self.filtered_data = filtered_data

        self.filtered_fft = scipy.fft.fft(filtered_data)

    def plot_data(self):

        plt.subplots(2, 2, sharex='col')

        # Raw + filtered data ----------------------------------------------------------------------------------------

        plt.subplot(2, 2, 1)
        plt.title("Raw and Filtered Data")
        plt.plot(self.x/self.fs, self.raw_data, label="{} sine wave(s)".format(self.data_freq), color='red')
        plt.legend()
        plt.ylabel("Amplitude")

        plt.subplot(2, 2, 3)
        plt.plot(self.x/self.fs, self.filtered_data, label=self.filter_details, color='black')
        plt.legend()
        plt.ylabel("Amplitude")
        plt.xlabel("Seconds")

        # FFT data ---------------------------------------------------------------------------------------------------

        xf = np.linspace(0.0, 1.0 / (2.0 * (1 / self.fs)), self.length // 2)

        plt.subplot(2, 2, 2)
        plt.title("Fast Fourier Transform Data")
        plt.plot(xf, 2.0 / self.length / 2 * np.abs(self.raw_fft[0:self.length // 2]),
                 color='black', label="{} sine wave(s)".format(self.data_freq))
        plt.ylabel("Power")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(xf, 2.0 / self.length / 2 * np.abs(self.filtered_fft[0:self.length // 2]),
                 color='red', label=self.filter_details)

        plt.ylabel("Power")
        plt.xlabel("Frequency (Hz)")
        plt.legend()


t = FilterTest(freqs=[1, 5, 9], filter_type="bandpass", low_f=3, high_f=7, filter_order=1)
