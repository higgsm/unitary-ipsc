# minimal_stim_analysis.py
# ANALYZE EVOKED SYNAPTIC CURRENTS WITH MINIMAL STIM
# Write results to file for each cell.

import os
import numpy as np
from scipy import optimize
import json
import matplotlib.pyplot as plt
import mh_vclamp as mh

# File paths
code_path = os.getcwd()
data_path = os.getcwd() + '/../Axograph/'
figure_path = os.getcwd() + '/../Figures/'
log_path = os.getcwd() + '/../Results/'

# Start log as empty dictionary
log = {}

# Load data
data = mh.load_vclamp(data_path)
file = data['filename']
times = data['times']
currents = data['currents']
dt = data['dt']
n_episodes = data['n episodes']
n_points = data['n points']
episodes = data['episodes']
episode_length = data['episode length']
print(file)
print(str(n_episodes) + ' episodes')
print('sampling interval = ' + str(dt) + ' s')
print('episode length = ' + str(episode_length) + ' s')
log['file'] = file
log['n episodes'] = n_episodes
log['sampling interval (s)'] = dt
log['episode length (s)'] = episode_length

# Function to return data index based on the sampling interval
def idx(t):
    return round(t / dt)

# Plot data
plot_episodes = data['episodes']
t_min_plot = 0.251
t_max_plot = 0.280
y_min = 0
y_max = 0
mh.plot_vclamp(data, plot_episodes, t_min_plot, t_max_plot, y_min, y_max, figure_path)

# Identify glitched episodes (positive glitches only)
t_min = 0.220
t_max = 0.250
segs = [current[idx(t_min): idx(t_max)] for current in currents]
maxima = [max(seg) for seg in segs]
deviations = maxima - np.median(maxima)
threshold = np.median(maxima) + 4 * np.std(deviations)
bad1 = [episode for episode in episodes if maxima[episode] > threshold]

t_min = 0.251
t_max = 0.290
segs = [current[idx(t_min): idx(t_max)] for current in currents]
maxima = [max(seg) for seg in segs]
deviations = maxima - np.median(maxima)
threshold = np.median(maxima) + 10 * np.std(deviations)
bad2 = [episode for episode in episodes if maxima[episode] > threshold]

bad_episodes = [e for e in episodes if (e in bad1) or (e in bad2)]
good_episodes = [e for e in episodes if e not in bad_episodes]

print('bad episodes:')
print(bad_episodes)
log['bad episodes'] = bad_episodes

# Measure mean current and access resistance
pulse_amplitude = 0.005
pulse_time = 0.0025
access_data = mh.measure_access(data, pulse_amplitude, pulse_time)

# Subtract baselines and plot good episodes
t_min_baseline = 0.010
t_max_baseline = 0.250
mh.baseline_subtract(data, t_min_baseline, t_max_baseline)

# Plot good episodes
t_min = 0.249
t_max = 0.260
data_min = min([min(segs[e]) for e in good_episodes])
data_max = max([max(segs[e]) for e in good_episodes])
data_range = data_max - data_min
plot_min = data_min - 0.05 * data_range
plot_max = data_max + 0.05 * data_range
mh.plot_vclamp(data, good_episodes, t_min, t_max, plot_min, plot_max, figure_path)

# Plot good responses sequentially
plot_episodes = good_episodes
stim_time = 0.250
t_min_plot = stim_time + 0.002
t_max_plot = stim_time + 0.040
spacer_length = 0.02
seg_length = t_max_plot - t_min_plot
for i in range(len(plot_episodes)):
    episode = plot_episodes[i]
    seg = mh.segment(currents[episode], dt, t_min_plot, t_max_plot)
    seg_x = []
    t0 = episode * (seg_length + spacer_length)
    for j in range(len(seg)):
        seg_x.append((t0 + j * dt) / (seg_length + spacer_length))
    plt.plot(seg_x, seg, 'k', linewidth = 0.75)
plt.grid()
plt.xlabel('trial', fontsize = 14)
plt.ylabel('current (pA)', fontsize = 14)
plt.show()

# Select episodes for analysis
# USE ONLY GOOD EPISODES
first_selected_episode = int(input('First selected episode: '))
last_selected_episode = int(input('Last selected episode: '))
selected_episodes = [e for e in good_episodes if (e >= first_selected_episode) and (e <= last_selected_episode)]

# Find synaptic currents
t_min = 0.012
psc_end = 0.030
t_max = episode_length - psc_end
baseline_start = -0.002
sigma = 0.0002
filter_times = np.arange(-5 * sigma, 5 * sigma, dt)
filter_values = np.exp(-filter_times**2 / (2 * sigma**2))
filter_values /= filter_values.sum()
start_point = round(len(filter_values) / 2)
smoothed_currents = []
diffs = []
for episode in episodes:
    current = currents[episode]
    padded_current = np.append(current, np.full(len(filter_values), current[-1]))
    smoothed_current = np.convolve(padded_current, filter_values)
    smoothed_current = smoothed_current[start_point:start_point + n_points]
    diff = np.diff(smoothed_current)
    diff /= np.median(np.abs(diff))
    smoothed_currents.append(smoothed_current)
    diffs.append(diff)

# Find PSCs in each episode
psc_data = mh.find_all_pscs(data, t_min, t_max)
n_pscs = psc_data['n PSCs']
psc_times_raw = psc_data['PSC times (s)']
psc_amplitudes = psc_data['PSC amplitudes (pA)']
psc_times = []
for episode in episodes:
    psc_times.append([t for t in psc_times_raw[episode] if (t < stim_time - 0.001) or (t > stim_time + 0.0018)])
    #psc_times.append([t for t in psc_times_raw[episode]])

# Plot detected PSC times for selected episodes
# Re-select episodes if necessary
t_min_plot = stim_time + 0.001
t_max_plot = stim_time + 0.010
done = False
while not done:
    plot_episodes = []
    psc_times_selected = []
    for episode in selected_episodes:
        for t in psc_times[episode]:
            if (t >= t_min_plot) and (t < t_max_plot):
                plot_episodes.append(episode)
                psc_times_selected.append(t)           
    plt.plot(psc_times_selected, plot_episodes, 'k', marker = '.', markersize = 6, linestyle = 'none')
    plt.xlim([stim_time, stim_time + 0.010])
    plt.xlabel('time (s)', fontsize = 12)
    plt.ylabel('episode', fontsize = 12)
    plt.grid()
    plt.show()

    accept = input('Accept these episodes? (y/n) ')
    if accept == 'n':
        first_selected_episode = int(input('First selected episode: '))
        last_selected_episode = int(input('Last selected episode: '))
        selected_episodes = [e for e in good_episodes if (e >= first_selected_episode) and (e <= last_selected_episode)]
    else:
        done = True
log['selected episodes'] = selected_episodes

# Find latency band, categorize responses into successes/failures, clean/dirty
def distance(x, center):
    return 1 - np.exp(-abs(x - center) / 0.0003)
def psc_times_distance(center):
    total = 0
    for t in psc_times_selected:
        total += distance(t, center)
    return total

done = False
while not done:

    # Set latency window
    guess = float(input('Approximate band center: '))
    window_width = float(input('Window width (default = 0.001): ').strip() or '0.001')
    lower_bound = guess - 0.001
    upper_bound = guess + 0.001
    result = optimize.minimize(fun = psc_times_distance, bounds = [(lower_bound, upper_bound)], x0 = guess)
    window_center = result['x'][0]
    window_start = window_center - window_width / 2
    window_end = window_center + window_width / 2
    frame_start = window_start - 0.030
    frame_end = window_end + 0.030

    # Categorize responses 
    psc_times_window = []
    successes = []
    failures = []
    clean_successes = []
    clean_failures = []
    cleans = []
    dirties = []
    clean_success_latencies = []
    for episode in selected_episodes:
        psc_times_w = [t for t in psc_times[episode] if t >= window_start and t < window_end]
        psc_times_f = [t for t in psc_times[episode] if t >= frame_start and t < frame_end]
        psc_times_window.append(psc_times_w)
        if len(psc_times_w) > 0:
            successes.append(episode)
            if len(psc_times_f) == 1:
                clean_successes.append(episode)
                clean_success_latencies.append(psc_times_w[0] - stim_time)
        else:
            failures.append(episode)
            if len(psc_times_f) == 0:
                clean_failures.append(episode)
        if len(psc_times_f) == len(psc_times_w):
            cleans.append(episode)
        else:
            dirties.append(episode)
    n_episodes_analyzed = len(selected_episodes)
    n_successes = len(successes)
    n_failures = len(failures)
    n_clean_successes = len(clean_successes)
    n_clean_failures = len(clean_failures)
    success_probability = n_successes / len(selected_episodes)
    clean_probability = n_clean_successes / n_successes
    clean_success_latency_mean = np.mean(clean_success_latencies) * 1000
    latency_window_start = 1000 * (window_start - stim_time)
    latency_window_end = 1000 * (window_end - stim_time)

    # Plot with clean responses green, dirty responses red (whole frame width)
    t_min_plot = frame_start
    t_max_plot = frame_end
    psc_times_clean = [t for episode in clean_successes for t in psc_times[episode] if (t >= t_min_plot) and (t < t_max_plot)]
    psc_episodes_clean = [episode for episode in clean_successes for t in psc_times[episode] if (t >= t_min_plot) and (t < t_max_plot)]
    psc_times_dirty = [t for episode in dirties for t in psc_times[episode] if (t >= t_min_plot) and (t < t_max_plot)]
    psc_episodes_dirty = [episode for episode in dirties for t in psc_times[episode] if (t >= t_min_plot) and (t < t_max_plot)]
    plt.plot(psc_times_clean, psc_episodes_clean, 'green', marker = '.', markersize = 6, linestyle = 'none')
    plt.plot(psc_times_dirty, psc_episodes_dirty, 'red', marker = '.', markersize = 6, linestyle = 'none')
    plt.axvline(window_start, color = 'red', linewidth = 0.75)
    plt.axvline(window_end, color = 'red', linewidth = 0.75)
    plt.xlim([frame_start, frame_end])
    plt.grid()
    plt.xlabel('time (s)', fontsize = 12)
    plt.ylabel('episode', fontsize = 12)
    plt.show()

    # Does this look okay?
    accept = input('Accept this latency band? (y/n) ')
    if accept == 'y':
        done = True

print(str(n_episodes_analyzed) + ' episodes analyzed')
print('latency window: {:.2f} to {:.2f} ms'.format(latency_window_start, latency_window_end))
print('mean clean success latency = {:.2f} ms'.format(clean_success_latency_mean))
print(str(n_successes) + ' successes, ' + str(n_clean_successes) + ' clean')
print(str(n_failures) + ' failures, ' + str(n_clean_failures) + ' clean')
print('success probability = {:.3f}'.format(success_probability))
print('clean probability for successes = {:.3f}'.format(clean_probability))
log['n episodes analyzed'] = n_episodes_analyzed
log['mean clean success latency'] = clean_success_latency_mean
log['n successes'] = n_successes
log['n failures'] = n_failures
log['success probability'] = success_probability
log['clean probability for successes'] = clean_probability

# Plot clean successes and failures
t_min = stim_time + 0.001
t_max = frame_end
plot_episodes = []
max_episodes = len(episodes)
plot_episodes = clean_successes + clean_failures
plot_episodes.sort()
if len(plot_episodes) > max_episodes:
    plot_episodes = plot_episodes[:max_episodes]
times_from_stim = [(t - stim_time) * 1000 for t in times]
for episode in plot_episodes:
    x_values = mh.segment(times_from_stim, dt, t_min, t_max)
    y_values = mh.segment(currents[episode], dt, t_min, t_max)
    if episode in clean_successes:
        plt.plot(x_values, y_values, 'k', linewidth = 0.75)
    if episode in clean_failures:
        plt.plot(x_values, y_values, 'gray', linewidth = 0.75)
plt.xlabel('time (ms)', fontsize = 12)
plt.ylabel('current (pA)', fontsize = 12)
plt.tight_layout()
plt.show()

# Plot clean successes and clean failures sequentially
t_min = min([stim_time + 0.002, window_start])
spacer_length = 0.02
seg_length = t_max - t_min
for i in range(len(plot_episodes)):
    episode = plot_episodes[i]
    seg = mh.segment(currents[episode], dt, t_min, t_max)
    seg_x = []
    for j in range(len(seg)):
        seg_x.append(i + (j * dt) / (seg_length + spacer_length))
    if episode in clean_successes:
        plt.plot(seg_x, seg, 'r', linewidth = 0.75)
    else:
        plt.plot(seg_x, seg, 'b', linewidth = 0.75)
plt.xlabel('response number, clean only', fontsize = 12)
plt.ylabel('current (pA)', fontsize = 12)
plt.tight_layout()
plt.show()

# Subtract average clean failure if desired
t_min = stim_time + 0.001
do_subtract = input('Subtract average clean failure? (y/n): ')
if do_subtract == 'y':
    avg_clean_failure = np.average(currents[clean_failures], axis = 0)
    cc = []
    for episode in episodes:
        cc.append(currents[episode] - avg_clean_failure)
    currents_cor = np.asarray(cc)
    plot_title = 'Clean successes, corrected'
else:
    currents_cor = currents.copy()
    plot_title = 'Clean successes, uncorrected'
x_data = mh.segment(times, dt, t_min, t_max)
for episode in clean_successes:
    y_data = mh.segment(currents_cor[episode], dt, t_min, t_max)
    plt.plot(x_data, y_data, 'k', linewidth = 0.75)
plt.title(plot_title, fontsize = 14)
plt.xlabel('time (s)', fontsize = 12)
plt.ylabel('current (pA)', fontsize = 12)
plt.show()

# Measure peak amplitude of each clean success
filter_sd = 0.0002
filter_times = np.arange(-5 * filter_sd, 5 * filter_sd, dt)
filter_values = np.exp(-filter_times**2 / (2 * sigma**2))
filter_values /= filter_values.sum()
start_point = round(len(filter_values) / 2)
smoothed_segments = []
peak_currents = []
for episode in episodes:
    current = currents_cor[episode]
    padded_current = np.append(current, np.full(len(filter_values), current[-1]))
    smoothed_current = np.convolve(padded_current, filter_values)
    smoothed_current = smoothed_current[start_point:start_point + n_points]
    smoothed_segment = mh.segment(smoothed_current, dt, window_start, frame_end)
    smoothed_segments.append(smoothed_segment)
    peak_currents.append(min(smoothed_segment))
peak_currents_clean = []
for episode in clean_successes:
    peak_currents_clean.append(peak_currents[episode])
peak_mean = np.mean(peak_currents_clean)
peak_sd = np.std(peak_currents_clean)

plot_min = min(peak_currents_clean) * 1.1
plt.plot(clean_successes, peak_currents_clean, 'k', marker = '.', markersize = 10, linestyle = 'none')
plt.ylim([plot_min, 0])
plt.xlabel('episode', fontsize = 12)
plt.ylabel('peak amplitude (pA)', fontsize = 12)
plt.show()
print('mean peak = {:.1f} pA (SD {:.1f})'.format(peak_mean, peak_sd))
log['mean peak (pA)'] = peak_mean
log['peak SD (pA)'] = peak_sd

# Align and average clean successes
t_rel_min = -0.0015
t_rel_max = 0.030
mean_psc_times = list(np.arange(t_rel_min, t_rel_max, dt))
pscs = []
for e in range(len(clean_successes)):
    episode = clean_successes[e]
    psc_times_window = [t for t in psc_times[episode] if t >= window_start and t < window_end]
    t = stim_time + clean_success_latencies[e]
    idx_min = round((t + t_rel_min) / dt)
    idx_max = idx_min + len(mean_psc_times)
    psc = currents_cor[episode].copy()[idx_min: idx_max];
    pscs.append(psc)
mean_psc = np.mean(pscs, axis = 0)

# Fit with single-exponential decay
t0_guess = -0.0005
a_guess = min(mean_psc)
k_rise_guess = 3000
k_d_guess = 100
fit = mh.fit_psc_single(mean_psc_times, mean_psc, t_rel_min, t0_guess, a_guess, k_rise_guess, k_d_guess)
plt.plot(mean_psc_times, mean_psc, 'k')
plt.plot(fit['fit times'], fit['fit values'], 'r')
plt.xlabel('time (s)', fontsize = 12)
plt.ylabel('current (pA)', fontsize = 12)
plt.show()
print('Fit with single-exponential decay:')
print('tau_r = {:.2f}'.format(fit['tau(r)']) + ' ms')
print('tau_d = {:.2f}'.format(fit['tau(d)']) + ' ms')
log['fit 1 tau(r) (ms)'] = fit['tau(r)']
log['fit 1 tau(d) (ms)'] = fit['tau(d)']

# Save results for this data file in case the double-exponential fit crashes
output_path = log_path + file + '_Results.txt'
with open(output_path, 'w') as f:
    f.write(json.dumps(log))

# Fit with double-exponential decay
t0_guess = -0.0005
a_fast_guess = min(mean_psc)
a_slow_guess = min(mean_psc)
k_rise_guess = 3000
k_d1_guess = 300
k_d2_guess = 30
fit = mh.fit_psc(mean_psc_times, mean_psc, t_rel_min, t0_guess, a_fast_guess, a_slow_guess, k_rise_guess, k_d1_guess, k_d2_guess)

plt.plot(mean_psc_times, mean_psc, 'k')
plt.plot(fit['fit times'], fit['fit values'], 'r')
plt.xlabel('time (s)', fontsize = 12)
plt.ylabel('current (pA)', fontsize = 12)
plt.show()

print('Fit with double-exponential decay:')
if type(fit['tau(r)']) == np.float64:
    print('tau_r = {:.2f}'.format(fit['tau(r)']) + ' ms')
    print('tau_f = {:.2f}'.format(fit['tau(f)']) + ' ms')
    print('tau_s = {:.2f}'.format(fit['tau(s)']) + ' ms')
    print('fraction fast = {:.3f}'.format(fit['fraction fast']))
    print('weighted tau_d = {:.2f}'.format(fit['tau(w)']) + ' ms')
else:
    print('ERROR')
    
log['fit 2 tau_r (ms)'] = fit['tau(r)']
log['fit 2 tau_f (ms)'] = fit['tau(f)']
log['fit 2 tau_s (ms)'] = fit['tau(s)']
log['fit 2 fraction fast'] = fit['fraction fast']
log['fit 2 weighted tau(d) (ms)'] = fit['tau(w)']

# Add any comments to the results (log)
comments = input('Any comments?')
log['comments'] = comments

# Save results for this data file
output_path = log_path + file + '_Results.txt'
with open(output_path, 'w') as f:
    f.write(json.dumps(log))
