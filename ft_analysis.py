# ft_analysis.py
# DETERMINE FIRING FREQUENCY-TIME CURVES FOR CURRENT STEP EXPERIMENT

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
step_currents = [-20, -10, 0, 10, 20]
steps_per_cycle = len(step_currents)
step_on_time = 5
step_off_time = 10
v_correction = 0.010
spike_detection_level = -0.02
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

# Combine the data in a dictionary
data = {}
data['file'] = files[file_number]
data['times'] = times
data['voltages'] = voltages
data['dt'] = dt
data['n episodes'] = n_episodes
data['episodes'] = episodes
data['episode length'] = episode_length

# Function to return data index based on the sampling interval
def idx(t):
    return round(t / dt)

# Plot all the data
plt.ioff()
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

# Find spikes
spike_times = [mh.find_spike_times(vs, dt, spike_detection_level, min_interspike_interval) for vs in voltages]
spike_counts = [len(ts) for ts in spike_times]

# Plot spike count for each episode
plt.plot(spike_counts, 'k', marker = '.', linestyle = 'none')
plt.xlabel('episode')
plt.ylabel('spikes')
plt.show()

# Choose range of episodes to analyze
first_episode = int(input('Start with episode: '))
last_episode = int(input('End with episode: '))
good_eps = list(range(first_episode, last_episode + 1))

# Separate spike times into baseline, step, and post-step
spike_times_base = []
spike_times_step = []
spike_times_post = []
for episode in episodes:
    spike_times_base.append([t for t in spike_times[episode] if t < step_on_time])
    spike_times_step.append([t for t in spike_times[episode] if (t >= step_on_time) and (t < step_off_time)])
    spike_times_post.append([t for t in spike_times[episode] if t >= step_off_time])

# Find instantaneous firing rates for baseline, step, and post-step
inst_rates_base = [list(1 / np.diff(ts)) for ts in spike_times_base]
inst_rates_step = [list(1 / np.diff(ts)) for ts in spike_times_step]
inst_rates_post = [list(1 / np.diff(ts)) for ts in spike_times_post]

# Find ISI midpoint times for baseline, step, and post-step
def list_means(lst):
    return [(lst[i] + lst[i + 1]) / 2 for i in range(len(lst) - 1)]
    
mid_times_base = [list_means(ts) for ts in spike_times_base]
mid_times_step = [list_means(ts) for ts in spike_times_step]
mid_times_post = [list_means(ts) for ts in spike_times_post]

# Start a figure (interactive mode on)
w, h = 7.15, 7.15
plt.ion()
fig = plt.figure(figsize = (w, h))

# Collect f-t data for each step level, good episodes only
plots_xvals = []
plots_yvals = []
for level in range(steps_per_cycle):
    plot_eps = [ep for ep in good_eps if ep % steps_per_cycle == level]
    plot_xvals = []
    plot_yvals = []
    for ep in plot_eps:
        for t in mid_times_base[ep]:
            plot_xvals.append(t)
        for t in mid_times_step[ep]:
            plot_xvals.append(t)
        for t in mid_times_post[ep]:
            plot_xvals.append(t)
        for r in inst_rates_base[ep]:
            plot_yvals.append(r)
        for r in inst_rates_step[ep]:
            plot_yvals.append(r)
        for r in inst_rates_post[ep]:
            plot_yvals.append(r)
    plots_xvals.append(plot_xvals)
    plots_yvals.append(plot_yvals)

# Save f-t data
results = {}
results['ISI midpoint times baseline (s)'] = mid_times_base
results['ISI midpoint times during step (s)'] = mid_times_step
results['instantaneous rates baseline (/s)'] = inst_rates_base
results['instantaneous rates during step (/s)'] = inst_rates_step
output_path = results_path + file + '_Results.txt'
with open(output_path, 'w') as f:
    f.write(json.dumps(results))

# Plot traces for the larger positive and negative steps
ax1 = fig.add_axes([0.12, 0.83, 0.35, 0.12])
ax1.plot(times, voltages[4], 'k', linewidth = 0.75)
ax1.spines[['right', 'top']].set_visible(False)
plt.xlabel('time (s)')
plt.ylabel('V')

ax1b = fig.add_axes([0.12, 0.96, 0.35, 0.03])
ax1b.plot([-5,0,0,5,5,10], [0,0,20,20,0,0], 'k', linewidth = 1)
ax1b.spines[['right', 'top', 'bottom']].set_visible(False)
plt.xticks([])

ax2 = fig.add_axes([0.12, 0.58, 0.35, 0.12])
ax2.plot(times, voltages[0], 'k', linewidth = 0.75)
ax2.spines[['right', 'top']].set_visible(False)
plt.xlabel('time (s)')
plt.ylabel('V')

ax2b = fig.add_axes([0.12, 0.71, 0.35, 0.03])
ax2b.plot([-5,0,0,5,5,10], [0,0,-20,-20,0,0], 'k', linewidth = 1)
ax2b.spines[['right', 'top', 'bottom']].set_visible(False)
plt.xticks([])

# Plot the rates for the larger positive and negative steps
ax3 = fig.add_axes([0.62, 0.83, 0.35, 0.12])
ax3.plot(plots_xvals[4], plots_yvals[4], 'k', marker = '.', linestyle = 'none')
ax3.spines[['right', 'top']].set_visible(False)
plt.xlabel('time (s)')
plt.ylabel('rate (/s)')

ax4 = fig.add_axes([0.62, 0.58, 0.35, 0.12])
ax4.plot(plots_xvals[0], plots_yvals[0], 'k', marker = '.', linestyle = 'none')
ax4.spines[['right', 'top']].set_visible(False)
plt.xlabel('time (s)')
plt.ylabel('rate (/s)')

plt.pause(0.0001)
plt.savefig(figure_path + file + "_ft.pdf", format = "pdf", bbox_inches = "tight")


    
