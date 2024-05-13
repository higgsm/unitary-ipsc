# mean_trajectories.py
# DETERMINE THE MEAN INTER-SPIKE MEMBRANE POTENTIAL TRAJECTORY
# IN A SPECIFIED TIME RANGE (BEFORE THE STIMULUS)
# SAMPLING EACH TRAJECTORY AT 10001 POINTS, INCLUDING BOTH SPIKE TIMES

import os
import axographio
import numpy as np
import matplotlib.pyplot as plt
import json
import mh_functions as mh
import math

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
n_samples = 10001

# Collect the data file names
files = [name[:-5] for name in sorted(os.listdir(data_path)) if name.endswith('.axgd')]
n_cells = len(files)
count = 0
for file in files:
    print(str(count) + '\t' + file)
    count += 1

# Start a sum for all the trajectories
t_trajectory_sum = [0.0 for i in range(n_samples)]
v_trajectory_sum = [0.0 for i in range(n_samples)]

# RUN FOR ALL FILES
for file in files:
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
    data['file'] = file
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
    plt.ioff()
    for vs in voltages:
        plt.plot(times, vs, 'k', linewidth = 0.5)
    plt.xlabel('time (s)')
    plt.ylabel('V')
    plt.show()

    # Choose range of episodes to analyze
    first_episode = int(input('Start with episode: '))
    last_episode = int(input('End with episode: '))
    good_episodes = list(range(first_episode, last_episode + 1))

    # Find spikes
    spike_times = [mh.find_spike_times(vs, dt, spike_detection_level, min_interspike_interval) for vs in voltages]

    # Collect and sum inter-spike membrane potential trajectories
    count = 0
    for episode in good_episodes:
        for spikenum in range(len(spike_times[episode]) - 1):
            t0 = spike_times[episode][spikenum]
            t1 = spike_times[episode][spikenum + 1]
            if t1 < stim_time:
                delta = (t1 - t0) / (n_samples - 1)
                sample_times = [t0 + delta * point for point in range(n_samples)]
                sample_points = [round(t / dt) for t in sample_times]
                sample_vs = [voltages[episode][point] for point in sample_points]
                count += 1
                if (count == 1):
                    t_sum = [t - t0 for t in sample_times]
                    v_sum = sample_vs
                else:
                    t_sum = [t_sum[i] + sample_times[i] - t0 for i in range(n_samples)]
                    v_sum = [v_sum[i] + sample_vs[i] for i in range(n_samples)]
    t_avg = [t_sum[i] / count for i in range(n_samples)]
    v_avg = [1000 * v_sum[i] / count for i in range(n_samples)]

    t_trajectory_sum = [t_trajectory_sum[i] + t_avg[i] for i in range(n_samples)]
    v_trajectory_sum = [v_trajectory_sum[i] + v_avg[i] for i in range(n_samples)]

    # Save mean trajectory
    results = {
        'times (s)': t_avg,
        'voltages (mV)': v_avg
        }
    output_path = results_path + file + '_MeanTrajectory.txt'
    with open(output_path, 'w') as f:
        f.write(json.dumps(results))

    # Plot
    plt.plot(t_avg, v_avg, 'k')
    plt.show()

# Grand average trajectory
t_grand = [t / len(files) for t in t_trajectory_sum]
v_grand = [v / len(files) for v in v_trajectory_sum]
results = {
    'times (s)': t_grand,
    'voltages (mV)': v_grand
    }
df = pd.DataFrame(results)
df.to_csv(results_path + 'GrandMeanTrajectory.txt', sep = '\t')

# Plot
plt.plot(t_avg, v_grand, 'k')
plt.title('Average trajectory from ' + str(len(files)) + ' cells')
plt.xlabel('time (s)')
plt.ylabel('mV')
plt.show()



