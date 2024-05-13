# ipsp_reversal.py
# DETERMINE REVERSAL POTENTIAL FOR SYNAPTIC POTENTIALS
# BY FINDING SLOPE OF IPSP VERSUS MEMBRANE POTENTIAL

import os
import axographio
import numpy as np
import matplotlib.pyplot as plt
import json
import mh_iclamp as mh
import math
from scipy.optimize import curve_fit

# File paths
data_path = os.getcwd() + '/../Axograph/'
figure_path = os.getcwd() + '/../Figures/'
results_path = os.getcwd() + '/../Results/'

# Analysis parameters
stim_time = 2
v_correction = 0.010
spike_detection_level = -0.025
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
volts = np.asarray(axgd.data[1:])
volts += v_correction
dt = times[1] - times[0]
[n_episodes, n_points] = volts.shape
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
            if volts[episode][i] < bottom_clip:
                volts[episode][i] = bottom_clip
            if volts[episode][i] > top_clip:
                volts[episode][i] = top_clip

# Combine the data in a dictionary
data = {}
data['file'] = files[file_number]
data['times'] = times
data['voltages'] = volts
data['dt'] = dt
data['n episodes'] = n_episodes
data['episodes'] = episodes
data['episode length'] = episode_length

# Function to return data index based on the sampling interval
def idx(t):
    return round(t / dt)

# Remove 60 Hz from each voltage trace
response = input('Subtract 60 Hz oscillation? (y/n) ')
if response == 'y':
    print('Removing 60 Hz ...')
    voltages = []
    for v in volts:
        fit = mh.fit_sin60(times, v)
        sine_values = [fit['fitfunc no offset'](t) for t in times]
        voltages.append([v[i] - sine_values[i] for i in range(n_points)])
else:
    voltages = volts.copy()

# Plot all of the data
for vs in voltages:
    plt.plot(times, vs, 'k', linewidth = 0.5)
plt.xlabel('time (s)')
plt.ylabel('V')
plt.show()

# Plot the mean voltage for each episode
plt.plot([np.mean(vs) for vs in voltages], 'k', marker = '.', linestyle = 'none')
plt.xlabel('episode')
plt.ylabel('mean Vm (V)')
plt.show()

# Choose range of episodes to analyze
first_episode = int(input('Start with episode: '))
last_episode = int(input('End with episode: '))
good_eps = list(range(first_episode, last_episode + 1))

# Find spikes
spike_times = [mh.find_spike_times(vs, dt, spike_detection_level, min_interspike_interval) for vs in voltages]
for episode in episodes:
    x_data = spike_times[episode]
    y_data = [episode for t in spike_times[episode]]
    plt.plot(x_data, y_data, 'k', marker = '.', linestyle = 'none')
plt.title('Spike times')
plt.xlabel('time (s)')
plt.ylabel('episode')
plt.show()

# Select episodes with no spike for 10 ms before or after the stimulus
range_start = stim_time - 0.010
range_end = stim_time + 0.010
good_episodes = []
for episode in good_eps:
    if len([st for st in spike_times[episode] if (st >= range_start) and (st < range_end)]) == 0:
        good_episodes.append(episode)

# dV/dt, smoothed by convolution with Gaussian kernel (0.5 ms SD)
def pad(lst):
    result = lst.copy()
    result.append(result[-1])
    return result

v_slopes = [pad(list(np.diff(v) / dt)) for v in voltages]
kernel_sd = 0.0005
v_slopes_smooth = [mh.gauss_smooth(ss, dt, kernel_sd) for ss in v_slopes]

# Plot dV/dt for the good episodes
idx_start = idx(stim_time + 0.001)
idx_end = idx(stim_time + 0.010)
for episode in good_episodes:
    plt.plot(times[idx_start:idx_end], v_slopes_smooth[episode][idx_start:idx_end], 'k', linewidth = 0.5)
plt.xlabel('time (s)')
plt.ylabel('dV/dt (V/s)')
plt.show()

# Figure out where to measure the IPSP slopes
# Want the negative peak of the average slope for traces starting above the median voltage
start_idx = idx(stim_time + 0.002)
end_idx = idx(stim_time + 0.010)
start_voltages = [vs[start_idx] for vs in voltages]
v_slopes_good_high = [v_slopes_smooth[episode] for episode in good_episodes if start_voltages[episode] > np.median(start_voltages)]
v_slopes_good_high_avg = np.mean(v_slopes_good_high, axis = 0)
measure_idx = start_idx
most_negative_slope = v_slopes_good_high_avg[start_idx]
for i in range(start_idx + 1, end_idx):
    slope = v_slopes_good_high_avg[i]
    if slope < most_negative_slope:
        measure_idx = i
        most_negative_slope = slope
measure_time = measure_idx * dt

# Measure v and dV/dt for each good episode, subtracting the pre-stimulus dV/dt
vs = [voltages[episode][measure_idx] for episode in good_episodes]
ss = [v_slopes_smooth[episode][measure_idx] for episode in good_episodes]
ss_pre = []
ss_post = []
for e in range(len(good_episodes)):
    episode = good_episodes[e]
    base_start_idx = idx(stim_time - 0.002)
    base_end_idx = idx(stim_time)
    m, b = np.polyfit(times[base_start_idx:base_end_idx], voltages[episode][base_start_idx:base_end_idx], 1)
    ss_pre.append(m)
    ss_post.append(ss[e])
    ss[e] -= m
    
# Plot the part around the stimulus, with measurement time indicated
idx_start = idx(stim_time - 0.05)
idx_end = idx(stim_time + 0.05)
for episode in good_episodes:
    plt.plot(times[idx_start:idx_end], voltages[episode][idx_start:idx_end], 'k', linewidth = 0.5)
#plt.ylim([-0.085, -0.030])
plt.axvline(x = measure_time, color = 'gray', linestyle = '--')
plt.title(file)
plt.xlabel('time (s)')
plt.ylabel('V')
plt.savefig(figure_path + file + "_IPSPs.pdf", format = "pdf", bbox_inches = "tight")
plt.show()

# Linear fit for V < -60 mV, or 75th percentile of the voltages if that was higher
v_cutoff = max([-0.060, np.percentile(vs, 75)])
vs_selected = [vs[i] for i in range(len(vs)) if vs[i] < v_cutoff]
ss_selected = [ss[i] for i in range(len(vs)) if vs[i] < v_cutoff]
m, b = np.polyfit(vs_selected, ss_selected, 1)
e_rev = -b / m
v_min = min(vs_selected)
if v_min > e_rev:
    v_min = e_rev
v_max = max(vs_selected)
fit_min = m * v_min + b
fit_max = m * v_max + b
plt.plot(vs, ss_pre, 'blue', marker = '.', linestyle = 'none')
plt.plot(vs, ss_post, 'green', marker = '.', linestyle = 'none')
plt.plot(vs, ss, 'red', marker = '.', linestyle = 'none')
plt.plot([v_min, v_max], [fit_min, fit_max], 'k')
plt.axhline(y = 0, color = 'gray', linestyle = '--')
plt.axvline(x = e_rev, color = 'gray', linestyle = '--')
plt.xlabel('voltage (V)')
plt.ylabel('IPSP slope (V/s)')
text_x = v_min + 0.25 * (v_max - v_min)
plt.text(text_x, -1, 'Erev = {:.1f} mV'.format(e_rev * 1000), fontsize = 12)
plt.savefig(figure_path + file + "_IpspSlopes.pdf", format = "pdf", bbox_inches = "tight")
plt.show()
print('Erev = {:.1f} mV'.format(e_rev * 1000))

# Synaptic PRC
# Define prior ISI (isi0) as mean before the stimulus
stim_phases = []
zs = []
for episode in episodes:
    spike_times_before = [t for t in spike_times[episode] if t <= stim_time]
    spike_times_after = [t for t in spike_times[episode] if t > stim_time]
    isis_before = np.diff(spike_times_before)
    isi0 = np.mean(isis_before)
    isi1 = spike_times_after[0] - spike_times_before[-1]
    stim_phases.append((stim_time - spike_times_before[-1]) / isi0)
    zs.append((isi1 - isi0) / isi0)
plt.plot(stim_phases, zs, 'k', marker = '.', linestyle = 'none')
plt.axhline(y = 0, color = 'black')
plt.xlim([0, 1])
plt.xlabel('stimulus phase')
plt.ylabel('phase change (cycles)')
plt.savefig(figure_path + file + "_SynapticPRC.pdf", format = "pdf", bbox_inches = "tight")
plt.show()

    
