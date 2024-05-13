# prc.py
# DETERMINE PHASE RESETTING CURVE (PRC) FROM NOISE STIMULATION DATA

import matplotlib as mpl
import os
import struct
import axographio
import numpy as np
import matplotlib.pyplot as plt
import json
import mh_iclamp as mh
import math
import pandas as pd
from scipy.optimize import curve_fit
from sklearn import linear_model

# Set pdf.fonttype to 42 (TrueType) and font to Arial so I can edit the pdf
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

# File paths
cwd = os.getcwd()
stimulus_path = cwd + '/../Stimuli/'
data_path = cwd + '/../Axograph/TauVirus/'
figure_path = cwd + '/../Figures/'
results_path = cwd + '/../Results/PRC/TauVirus/'

# Analysis parameters
episode_length = 20
noise_start_time = 4
v_correction = 0.010
spike_detection_level = -0.020
min_interspike_interval = 0.005
top_clip = 0.030
bottom_clip = -0.090

# Collect the data file names
files = [name[:-5] for name in sorted(os.listdir(data_path)) if name.endswith('.axgd')]
n_cells = len(files)
count = 0
for file in files:
    print(str(count) + '\t' + file)
    count += 1

# Select a file
file_number = int(input('Select file number: '))
file = files[file_number]
print(file)

# Load data from the file
axgd = axographio.read(data_path + file + '.axgd')
times = np.asarray(axgd.data[0])
voltages = np.asarray(axgd.data[1:])
voltages += v_correction
dt = times[1] - times[0]
[n_episodes, n_points] = voltages.shape
episode_length = n_points * dt
episodes = list(range(n_episodes))
print(str(n_episodes) + ' episodes')
print('sampling interval = ' + str(dt) + ' s')
print('episode length = ' + str(episode_length) + ' s')

# Clip large glitches in voltage traces
response = input('Clip glitches? (y/n) ')
if response == 'y':
    print('Clipping glitches ...')
    for episode in episodes:
        for i in range(n_points):
            if voltages[episode][i] < bottom_clip:
                voltages[episode][i] = bottom_clip
            if voltages[episode][i] > top_clip:
                voltages[episode][i] = top_clip

# Plot selected data
plt.ioff()
done = False
while not done:
    episode = int(input('plot episode (-1 to stop): '))
    if episode in episodes:
        plt.plot(times, voltages[episode], 'k', linewidth = 0.5)
        plt.axhline(spike_detection_level)
        plt.xlabel('time (s)')
        plt.ylabel('V')
        plt.show()
    else:
        done = True

# Change spike detection level
response = input('Change spike detection level? (y/n)')
if response == 'y':
    spike_detection_level = float(input('Spike detection level: '))

# Find spike times
print('Finding spike times ...')
spike_times = [mh.find_spike_times(vs, dt, spike_detection_level, min_interspike_interval) for vs in voltages]

# Separate spike times from baseline and noise periods
spike_times_base = []
spike_times_noise = []
for episode in episodes:
    spike_times_base.append([t for t in spike_times[episode] if t < noise_start_time])
    spike_times_noise.append([t for t in spike_times[episode] if t >= noise_start_time])

# Mean spike rates and CV of inter-spike intervals for baseline and noise periods
spike_rates_base = [len(st) / noise_start_time for st in spike_times_base]
spike_rates_noise = [len(st) / (episode_length - noise_start_time) for st in spike_times_noise]
isis = [[ts[i] - ts[i - 1] for i in range(1, len(ts))] for ts in spike_times]
isis_base = [[ts[i] - ts[i - 1] for i in range(1, len(ts))] for ts in spike_times_base]
isis_noise = [[ts[i] - ts[i - 1] for i in range(1, len(ts))] for ts in spike_times_noise]
cvs_base = [np.std(isis) / np.mean(isis) for isis in isis_base]
cvs_noise = [np.std(isis) / np.mean(isis) for isis in isis_noise]

plt.plot(spike_rates_base, 'blue', marker = '.', linestyle = 'none')
plt.plot(spike_rates_noise, 'red', marker = '.', linestyle = 'none')
plt.ylim(bottom = 0)
plt.xlabel('episode')
plt.ylabel('spikes/s')
plt.show()

plt.plot(cvs_base, 'blue', marker = '.', linestyle = 'none')
plt.plot(cvs_noise, 'red', marker = '.', linestyle = 'none')
plt.ylim(bottom = 0)
plt.xlabel('episode')
plt.ylabel('CV(ISI)')
plt.show()

# Choose episodes to analyze
first_episode = -1
last_episode = -1
while first_episode not in episodes:
    first_episode = int(input('First episode to analyze: '))
while last_episode not in episodes:
    last_episode = int(input('Last episode to analyze: '))

# Plot ISIs
for episode in range(first_episode, last_episode + 1):
    plt.plot(spike_times[episode][1:], isis[episode], 'k', marker = '.', linestyle = 'none')
plt.ylim(bottom = 0)
plt.xlabel('time (s)')
plt.ylabel('ISI (s)')
plt.show()

# Load stimulus (50 pA SD) from binary file, partition into episodes
# NEED TO UNDO SCALE FACTOR USED TO IMPORT THE STIMULUS INTO AXOGRAPH
stimulus_file = 'long_noise_2ms_50pA.dat'
print('Loading ' + stimulus_file + ' ...')
with open(stimulus_path + stimulus_file, mode = 'rb') as f:
    binary_data = f.read()
n = int(len(binary_data) / 4)
scaled_stim = struct.unpack('f' * n, binary_data)
stim = [s / 3276.8 for s in scaled_stim]
n_episodes_stim = int(len(stim) / n_points)
stimuli = [stim[episode * n_points: (episode + 1) * n_points] for episode in range(n_episodes_stim)]

# Collect charge for subdivisions of two ISIs (noise periods only)
# Collect the corresponding ISIs in a single list
print('Computing first- and second-order PRC ...')
n_bins = 40
phases = list(np.arange(-1 + 0.5 / n_bins, 1, 1 / n_bins))
charges = []
isi_list = []
for episode in range(first_episode, last_episode + 1):
    n_spikes = len(spike_times_noise[episode])
    for spike in range(n_spikes - 2):
        tsp0 = spike_times_noise[episode][spike]
        tsp1 = spike_times_noise[episode][spike + 1]
        tsp2 = spike_times_noise[episode][spike + 2]
        isi0 = tsp1 - tsp0
        isi = tsp2 - tsp1
        w0 = isi0 / n_bins
        w = isi / n_bins
        chg = []
        for b in range(n_bins):
            t0 = tsp0 + b * w0
            t1 = tsp0 + (b + 1) * w0
            idx0 = round(t0 / dt)
            idx1 = round(t1 / dt)
            chg.append(sum(stimuli[episode][idx0:idx1]) * dt)
        for b in range(n_bins):
            t0 = tsp1 + b * w
            t1 = tsp1 + (b + 1) * w
            idx0 = round(t0 / dt)
            idx1 = round(t1 / dt)
            chg.append(sum(stimuli[episode][idx0:idx1]) * dt)
        charges.append(chg)
        isi_list.append(isi)
mean_isi = np.mean(isi_list)
        
# Multiple linear regression of ISI vs. charges
X = pd.DataFrame(charges, columns = ['X' + str(bin + 1) for bin in range(len(phases))])
y = pd.DataFrame(isi_list, columns = ['Y'])   
model = linear_model.LinearRegression()
model.fit(X = X, y = y)
zs = [-c / mean_isi for c in model.coef_[0]]

# Standard errors
N = len(X)
p = len(X.columns) + 1  # plus one because LinearRegression adds an intercept term
X_with_intercept = np.empty(shape = (N, p), dtype = float)
X_with_intercept[:, 0] = 1
X_with_intercept[:, 1:p] = X.values
beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y.values
y_hat = model.predict(X)
residuals = y.values - y_hat
residual_sum_of_squares = residuals.T @ residuals
sigma_squared_hat = residual_sum_of_squares[0, 0] / (N - p)
var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
z_errors = []
for p_ in range(1, p):
    z_errors.append(var_beta_hat[p_, p_] ** 0.5 / mean_isi)

# PLOT FIRST- AND SECOND-ORDER PRC WITH STANDARD ERRORS
plt.axhline(0, color = 'gray')
plt.axvline(0, color = 'gray')
plt.errorbar(phases, zs, yerr = z_errors, color = 'k', marker = '.')
plt.xlim([-1, 1])
plt.xlabel('phase')
plt.ylabel('cycles/pA-s')
plt.show()

# SAVE PRC DATA TO FILE
print('Saving PRC data ...')
results = {}
results['episodes analyzed'] = list(range(first_episode, last_episode + 1))
results['mean ISI (s)'] = mean_isi
results['phases'] = phases
results['PRC values'] = zs
results['standard errors'] = z_errors
with open(results_path + file + '_PRC.txt', 'w') as f:
    f.write(json.dumps(results))
