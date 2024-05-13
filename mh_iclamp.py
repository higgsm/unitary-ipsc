# mh_iclamp.py
# DEFINE ANALYSIS FUNCTIONS FOR CURRENT CLAMP DATA

import csv
import axographio
from scipy.optimize import curve_fit
from scipy import optimize
from statistics import mean
from statistics import stdev
from pylab import *
import matplotlib.pyplot as plt
import os
import seaborn as sns

# collect list of axgd files in specified directory
def list_axgd_files(data_path):
    print(data_path)
    files = []
    count = 0
    for x in sorted(os.listdir(data_path)):
        if x.endswith(".axgd"):
            file = x[0:-5]
            files.append(file)
            print(str(count) + ', ' + file)
            count += 1
    return files

# pull out a segment of regularly sampled data by time (assuming first point is t = 0)
def segment(list, dt, t_min, t_max):
    return list.copy()[round(t_min / dt): round(t_max / dt)]

# flatten a list by one level
def flatten(list):
    result = []
    for sublist in list:
        for item in sublist:
            result.append(item)
    return result

# Smooth by convolution with a Gaussian kernel
def gauss_smooth(data, dt, kernel_sd):
    filter_times = np.arange(-5 * kernel_sd, 5 * kernel_sd, dt)
    filter_values = np.exp(-filter_times**2 / (2 * kernel_sd**2))
    filter_values /= filter_values.sum()
    padded_data = np.append(data, np.full(len(filter_values), data[-1]))
    smoothed_data = np.convolve(padded_data, filter_values)
    start_point = round(len(filter_values) / 2)
    return smoothed_data[start_point: start_point + len(data)]

# Fit sine wave to data
def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

# Fit 60 Hz sine wave to data
def fit_sin60(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    guess_amp = np.std(yy) * 2.0**0.5
    guess_phase = 0.0
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, guess_phase, guess_offset])
    f = 60
    w = 2.0 * np.pi * f

    def sinfunc60(t, A, p, c):
        w = 2.0 * np.pi * 60
        return A * np.sin(w * t + p) + c
    popt, pcov = optimize.curve_fit(sinfunc60, tt, yy, p0=guess)
    A, p, c = popt
    fitfunc = lambda t: A * np.sin(w * t + p) + c
    fitfunc_no_offset = lambda t: A * np.sin(w * t + p)
    return {"amp": A, "phase": p, "offset": c, "fitfunc": fitfunc, "fitfunc no offset": fitfunc_no_offset, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

# Find spike times in voltage trace
def find_spike_times(vs, dt, detection_level, min_interval):
    spike_times = []
    last_spike_time = -min_interval
    for i in range(1, len(vs)):
        t = i * dt
        if (vs[i - 1] < detection_level) and (vs[i] >= detection_level) and (t - last_spike_time >= min_interval):
            spike_times.append(t)
            last_spike_time = t
    return spike_times

