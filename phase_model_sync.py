# phase_model_sync.py
# SIMULATE VTA DOPMAINE NEURON RESPONSES TO uIPSG AND SYNCHRONOUS uIPSGs

import os
import axographio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import math
import csv
import random

# Set pdf.fonttype to 42 (TrueType) and font to Arial so I can edit the pdf
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

# File paths
figure_path = os.getcwd() + '/../PythonFigures/'
results_path = os.getcwd() + '/../PythonSimResults/'
model_data_path = os.getcwd() + '/../PythonModelData/'

# Model parameters (with conductance in nS, or pA/mV)
phi_bin_width = 0.025
phi_start = 0.006
phi_body_end = 1 - 1.5 * phi_bin_width
phi_peak = 1 - 0.5 * phi_bin_width
phi_end = 0.999
fit_a = 0.5921
fit_alpha = 1.668
fit_beta = 0.1128
fit_k = 0.05637
z_peak = 0.1834
z_body_end = fit_a * math.exp(-(phi_body_end - phi_start) / fit_beta) * pow(phi_body_end - phi_start, fit_alpha - 1) +  fit_k * (phi_body_end - phi_start)

omega = 2
isi0 = 1 / omega
e_rev = -63
mean_unitary_peak = 116 / 67
tau_rise = 0.0005
tau_decay = 0.0079
dt = 0.0001

# PRC function
def z(phi):
    if phi < phi_start:
        result = 0
    if (phi >= phi_start) and (phi < phi_body_end):
        result = fit_a * math.exp(-(phi - phi_start) / fit_beta) * pow(phi - phi_start, fit_alpha - 1) + fit_k * (phi - phi_start)
    if (phi >= phi_body_end) and (phi < phi_peak):
        result = z_body_end + (z_peak - z_body_end) * (phi - phi_body_end) / (phi_peak - phi_body_end)
    if (phi >= phi_peak) and (phi < phi_end):
        result = z_peak * (phi_end - phi) / (phi_end - phi_peak)
    if (phi >= phi_end):
        result = 0
    return result

# Inter-spike voltage trajectory
traj_times = []
traj_voltages = []
count = 0
with open(model_data_path + 'GrandMeanTrajectory.txt', newline='') as f:
    reader = csv.reader(f, delimiter='\t', quotechar='|')
    for row in reader:
        if count > 0:
            traj_times.append(float(row[1]))
            traj_voltages.append(float(row[2]))
        count += 1

# Convert trajectory to a function of phase
def v(phi):
    if phi < 0:
        result = traj_voltages[0]
    if (phi >= 0) and (phi <= 1):
        idx = round(phi * (len(traj_voltages) - 1))
        result = traj_voltages[idx]
    if phi > 1:
        result = traj_voltages[-1]
    return result

# Phase model for an inhibitory conductance waveform
def model(gs, dt, phi0, omega):
    phases = [0.0 for g in gs]
    voltages = [0.0 for g in gs]
    spike_times = []
    phi = phi0
    phases[0] = phi
    voltages[0] = v(phi)
    for i in range(1, len(gs)):
        phi += dt * (omega + gs[i - 1] * (e_rev - v(phi)) * z(phi))
        if phi > 1:
            phi -= 1
            spike_times.append(i * dt)
        phases[i] = phi
    return {'phases': phases, 'spike times': spike_times}

# Function to make conductance stimulus
def make_gstim(length, dt, psg_times, psg_amplitudes, tau_rise, tau_decay):
    n = round(length / dt)
    impulse_train = [0.0 for i in range(n)]
    for i in range(len(psg_times)):
        idx = round(psg_times[i] / dt)
        impulse_train[idx] += psg_amplitudes[i]
    filter_times = np.arange(0, 10 * tau_decay, dt)
    filter_values_unscaled = [-math.exp(-t / tau_rise) + math.exp(-t / tau_decay) for t in filter_times]
    mx = max(filter_values_unscaled)
    filter_values = [v / mx for v in filter_values_unscaled]
    return np.convolve(filter_values, impulse_train)[:n]


# ----- FIGURE -----
w, h = 7.15, 7.15
plt.ion()
fig = plt.figure(figsize = (w, h))


# ROW 1, THE MODEL

# Col 1, PRC
r1c1 = fig.add_axes([0.11, 0.86, 0.2, 0.1])
plot_phases = np.arange(0, 1, 0.001)
plot_zs = [z(phi) for phi in plot_phases]
r1c1.plot(plot_phases, plot_zs, 'k', linewidth = 1.25)
r1c1.spines[['right', 'top']].set_visible(False)
plt.ylim([0, 0.2])
plt.title('PRC')
plt.xlabel('phase')
plt.ylabel('cycles/pA-s')

# Col 2, trajectory
r1c2 = fig.add_axes([0.42, 0.86, 0.2, 0.1])
traj_phases = [t / traj_times[-1] for t in traj_times]
r1c2.plot(traj_phases, traj_voltages, 'k', linewidth = 1.25)
r1c2.axhline(e_rev, color = 'black', linestyle = '--', linewidth = 1.25)
r1c2.spines[['right', 'top']].set_visible(False)
plt.title('Voltage trajectory')
plt.xlabel('phase')
plt.ylabel('mV')

# Col 3, sensitivity
sens = [z(phi) * (e_rev - v(phi)) for phi in traj_phases]
r1c3 = fig.add_axes([0.77, 0.86, 0.2, 0.1])
r1c3.plot(traj_phases, sens, 'k', linewidth = 1.25)
r1c3.axhline(0, color = 'black', linestyle = '--', linewidth = 1)
r1c3.spines[['right', 'top']].set_visible(False)
plt.title('PRC-GI')
plt.xlabel('phase')
plt.ylabel('cycles/nS-s')


# ----- SIMULATIONS -----

# 20 x uIPSG AT phases 0.2 and 0.8
phi0 = 0
input_phases = [0, 0.2, 0.8]
ipsg_amplitude = 20 * mean_unitary_peak
ipsg_amplitudes = [0, ipsg_amplitude, ipsg_amplitude]
n_runs = len(input_phases)
length = 1.2 * isi0
n_points = round(length / dt)
times = [i * dt for i in range(n_points)]
times_ms = [1000 * t for t in times]
stimuli = []
phases = []
spike_times = []
voltages = []
for r in range(n_runs):
    psg_times = [input_phases[r] * isi0]
    stim = make_gstim(length, dt, psg_times, [ipsg_amplitudes[r]], tau_rise, tau_decay)
    model_data = model(stim, dt, phi0, omega)
    stimuli.append([s / mean_unitary_peak for s in stim])
    phases.append(model_data['phases'])
    spike_times.append(model_data['spike times'])
    voltages.append([v(phi) for phi in model_data['phases']])


# PLOT STIMULI
r2c1 = fig.add_axes([0.11, 0.71, 0.25, 0.05])
r2c1.plot(times_ms, stimuli[0], 'gray', linewidth = 1)
r2c1.plot(times_ms, stimuli[1], 'black', linewidth = 1)
r2c1.spines[['right', 'top', 'bottom']].set_visible(False)
plt.xticks([])
plt.ylabel('G (x uIPSG)')

r2c2 = fig.add_axes([0.44, 0.71, 0.25, 0.05])
r2c2.plot(times_ms, stimuli[0], 'gray', linewidth = 1)
r2c2.plot(times_ms, stimuli[2], 'black', linewidth = 1)
r2c2.spines[['right', 'top', 'bottom']].set_visible(False)
plt.xticks([])

# PLOT PHASE TRAJECTORIES
r3c1 = fig.add_axes([0.11, 0.56, 0.25, 0.1])
r3c1.plot(times_ms, phases[0], 'gray', linewidth = 1)
r3c1.plot(times_ms, phases[1], 'black', linewidth = 1)
r3c1.spines[['right', 'top', 'bottom']].set_visible(False)
plt.xticks([])
plt.ylabel('phase')

r3c2 = fig.add_axes([0.44, 0.56, 0.25, 0.1])
r3c2.plot(times_ms, phases[0], 'gray', linewidth = 1)
r3c2.plot(times_ms, phases[2], 'black', linewidth = 1)
r3c2.spines[['right', 'top', 'bottom']].set_visible(False)
plt.xticks([])

# PLOT VOLTAGE TRAJECTORIES
r4c1 = fig.add_axes([0.11, 0.41, 0.25, 0.1])
r4c1.plot(times_ms, voltages[0], 'gray', linewidth = 1)
r4c1.plot(times_ms, voltages[1], 'black', linewidth = 1)
r4c1.spines[['right', 'top']].set_visible(False)
plt.xlabel('time (s)')
plt.ylabel('mV')

r4c2 = fig.add_axes([0.44, 0.41, 0.25, 0.1])
r4c2.plot(times_ms, voltages[0], 'gray', linewidth = 1)
r4c2.plot(times_ms, voltages[2], 'black', linewidth = 1)
r4c2.spines[['right', 'top']].set_visible(False)
plt.xlabel('time (s)')

plt.pause(0.0001)


# RESPONSE TO 1 x uIPSG AT PHASES 0 to 0.99 by 0.01
phi0 = 0
input_phases = [0.01 * i for i in range(100)]
ipsg_times = [phi * isi0 for phi in input_phases]
ipsg_amplitude = mean_unitary_peak
n_runs = len(input_phases)
length = 1.2 * isi0
n_points = round(length / dt)
times = [i * dt for i in range(n_points)]
stimuli = []
phases = []
spike_times = []
voltages = []
for r in range(n_runs):
    stim = make_gstim(length, dt, [ipsg_times[r]], [ipsg_amplitude], tau_rise, tau_decay)
    model_data = model(stim, dt, phi0, omega)
    spike_times.append(model_data['spike times'])
spike_delays_1 = [1000 * (spike_times[run][0] - isi0) for run in range(n_runs)]

# RESPONSE TO 20 x uIPSG AT PHASES 0 to 0.99 by 0.01
phi0 = 0
input_phases = [0.01 * i for i in range(100)]
ipsg_times = [phi * isi0 for phi in input_phases]
ipsg_amplitude = 20 * mean_unitary_peak
n_runs = len(input_phases)
length = 1.4 * isi0
n_points = round(length / dt)
times = [i * dt for i in range(n_points)]
stimuli = []
phases = []
spike_times = []
voltages = []
for r in range(n_runs):
    stim = make_gstim(length, dt, [ipsg_times[r]], [ipsg_amplitude], tau_rise, tau_decay)
    model_data = model(stim, dt, phi0, omega)
    spike_times.append(model_data['spike times'])
spike_delays_20 = [1000 * (spike_times[run][0] - isi0) for run in range(n_runs)]
norm_delays_20 = [d / 20 for d in spike_delays_20]


# Row 2, col 3: spike delays for one uIPSG and norm. delays for 20 uIPSGs
r2c3 = fig.add_axes([0.83, 0.61, 0.13, 0.1])
r2c3.plot(input_phases, spike_delays_1, 'k', marker = '.', markersize = 2, linestyle = 'none')
r2c3.plot(input_phases, norm_delays_20, 'gray', marker = '.', markersize = 2, linestyle = 'none')
r2c3.spines[['right', 'top']].set_visible(False)
plt.xlabel('input phase')
plt.ylabel('spike delay (ms)')

# Row 3, col 3: spike delays for 20 uIPSGs
r3c3 = fig.add_axes([0.83, 0.41, 0.13, 0.1])
r3c3.plot(input_phases, spike_delays_20, 'k', marker = '.', markersize = 2, linestyle = 'none')
r3c3.spines[['right', 'top']].set_visible(False)
plt.xlabel('input phase')
plt.ylabel('spike delay (ms)')

plt.pause(0.0001)


# PSTH FOR 1x, 10x, and 100x uIPSG -- EXAMPLE PHASE TRAJECTORIES
n_runs = 20
length = 1.4 * isi0
ipsg_time = 0.2 * isi0
start_phases = [i / n_runs for i in range(n_runs)]
n_points = round(length / dt)
times = [i * dt for i in range(n_points)]
times_ms = [1000 * t for t in times]

print('Running PSTH 1 ...')
ipsg_amplitude = mean_unitary_peak
stim = make_gstim(length, dt, [ipsg_time], [ipsg_amplitude], tau_rise, tau_decay)
phases = []
for r in range(n_runs):
    model_data = model(stim, dt, start_phases[r], omega)
    phases.append(model_data['phases'])
stim_1 = stim.copy()
phases_1 = phases.copy()

print('Running PSTH 2 ...')
ipsg_amplitude = 10 * mean_unitary_peak
stim = make_gstim(length, dt, [ipsg_time], [ipsg_amplitude], tau_rise, tau_decay)
phases = []
for r in range(n_runs):
    model_data = model(stim, dt, start_phases[r], omega)
    phases.append(model_data['phases'])
phases_10 = phases.copy()

print('Running PSTH 3 ...')
ipsg_amplitude = 100 * mean_unitary_peak
stim = make_gstim(length, dt, [ipsg_time], [ipsg_amplitude], tau_rise, tau_decay)
phases = []
for r in range(n_runs):
    model_data = model(stim, dt, start_phases[r], omega)
    phases.append(model_data['phases'])
phases_100 = phases.copy()

# ROW 5, PLOT IPSG IN 3 COLUMNS (scale doesn't matter; will label)
r5c1 = fig.add_axes([0.11, 0.3, 0.15, 0.03])
r5c1.plot(times_ms, stim_1, 'k', linewidth = 1)
r5c1.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
plt.xlim([0, 700])
plt.xticks([])
plt.yticks([])

r5c2 = fig.add_axes([0.33, 0.3, 0.15, 0.03])
r5c2.plot(times_ms, stim_1, 'k', linewidth = 1)
r5c2.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
plt.xlim([0, 700])
plt.xticks([])
plt.yticks([])

r5c3 = fig.add_axes([0.55, 0.3, 0.15, 0.03])
r5c3.plot(times_ms, stim_1, 'k', linewidth = 1)
r5c3.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
plt.xlim([0, 700])
plt.xticks([])
plt.yticks([])

# ROW 6: PLOT PHASE TRAJECTORIES FOR 1x, 10x, and 100x uIPSG
r6c1 = fig.add_axes([0.11, 0.2, 0.15, 0.08])
bns = [i for i in range(int(-1000 * ipsg_time), int(1000 * length))]
for phases in phases_1:
    r6c1.plot(times_ms, phases, 'k', linewidth = 0.5)
r6c1.spines[['right', 'top', 'bottom']].set_visible(False)
plt.xlim([0, 700])
plt.xticks([])
plt.ylabel('phase')

r6c2 = fig.add_axes([0.33, 0.2, 0.15, 0.08])
for phases in phases_10:
    r6c2.plot(times_ms, phases, 'k', linewidth = 0.5)
r6c2.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
plt.xlim([0, 700])
plt.xticks([])
plt.yticks([])

r6c3 = fig.add_axes([0.55, 0.2, 0.15, 0.08])
for phases in phases_100:
    r6c3.plot(times_ms, phases, 'k', linewidth = 0.5)
r6c3.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
plt.xlim([0, 700])
plt.xticks([])
plt.yticks([])

plt.pause(0.0001)


# PSTH FOR 1x, 10x, and 100x uIPSG -- ALL TRIALS (10000)
n_runs = 10000
start_phases = [i / n_runs for i in range(n_runs)]
n_points = round(length / dt)
times = [i * dt for i in range(n_points)]

ipsg_amplitude = mean_unitary_peak
stim = make_gstim(length, dt, [ipsg_time], [ipsg_amplitude], tau_rise, tau_decay)
spike_times = []
for r in range(n_runs):
    model_data = model(stim, dt, start_phases[r], omega)
    spike_times.append(model_data['spike times'])
psth_spike_times_1 = [1000 * (t - ipsg_time) for r in range(n_runs) for t in spike_times[r]]

ipsg_amplitude = 10 * mean_unitary_peak
stim = make_gstim(length, dt, [ipsg_time], [ipsg_amplitude], tau_rise, tau_decay)
spike_times = []
for r in range(n_runs):
    model_data = model(stim, dt, start_phases[r], omega)
    spike_times.append(model_data['spike times'])
psth_spike_times_10 = [1000 * (t - ipsg_time) for r in range(n_runs) for t in spike_times[r]]

ipsg_amplitude = 100 * mean_unitary_peak
stim = make_gstim(length, dt, [ipsg_time], [ipsg_amplitude], tau_rise, tau_decay)
spike_times = []
for r in range(n_runs):
    model_data = model(stim, dt, start_phases[r], omega)
    spike_times.append(model_data['spike times'])
psth_spike_times_100 = [1000 * (t - ipsg_time) for r in range(n_runs) for t in spike_times[r]]

# FIND PAUSE DURATION FOR A RANGE OF IPSG AMPLITUDES
# 1 TO 100 x uIPSG, 2500 runs each
n_runs = 2500
ipsg_scales = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_sets = len(ipsg_scales)
ipsg_amplitudes = [scale * mean_unitary_peak for scale in ipsg_scales]
start_phases = [i / n_runs for i in range(n_runs)]
pause_durations = []
for set in range(n_sets):
    print('running set ' + str(set) + ' ...')
    stim = make_gstim(length, dt, [ipsg_time], [ipsg_amplitudes[set]], tau_rise, tau_decay)
    spike_times = []
    for r in range(n_runs):
        model_data = model(stim, dt, start_phases[r], omega)
        spike_times.append(model_data['spike times'])
    psth_spike_times = [1000 * (t - ipsg_time) for r in range(n_runs) for t in spike_times[r]]
    bin_width = 1
    bns = list(np.arange(-1000 * ipsg_time, 1000 * (length - ipsg_time), bin_width))
    counts, hbins = np.histogram(psth_spike_times, bns)
    expected_count = omega * n_runs * bin_width / 1000
    pause_start_found = False
    pause_end_found = False
    i = 0
    while (i < len(counts)) and (not pause_start_found):
        if counts[i] < 0.5 * expected_count:
            pause_start = bns[i]
            pause_start_found = True
        i += 1
    while (i < len(counts)) and (not pause_end_found):
        if counts[i] > 0.5 * expected_count:
            pause_end = bns[i]
            pause_end_found = True
        i += 1
    pause_durations.append(pause_end - pause_start)
    

# ROW 5, COL 4: PAUSE DURATION VS. NUMBER OF uIPSGs
r5c4 = fig.add_axes([0.82, 0.12, 0.15, 0.15])
r5c4.plot(ipsg_scales, pause_durations, 'k', marker = '.', markersize = 3, linestyle = 'none')
r5c4.spines[['right', 'top']].set_visible(False)
plt.xlabel('number of uIPSGs')
plt.ylabel('pause duration (ms)')


# ROW 7: PLOT PSTH FOR 1x, 10x, and 100x uIPSG
r7c1 = fig.add_axes([0.11, 0.07, 0.15, 0.1])
r7c1.hist(psth_spike_times_1, bins = bns, color = 'black')
r7c1.spines[['right', 'top']].set_visible(False)
plt.xlim([-100, 600])
plt.ylabel('spikes/s')

r7c2 = fig.add_axes([0.33, 0.07, 0.15, 0.1])
r7c2.hist(psth_spike_times_10, bins = bns, color = 'black')
r7c2.spines[['right', 'top']].set_visible(False)
plt.xlim([-100, 600])
plt.xlabel('time from IPSG onset (ms)')

r7c3 = fig.add_axes([0.55, 0.07, 0.15, 0.1])
r7c3.hist(psth_spike_times_100, bins = bns, color = 'black')
r7c3.spines[['right', 'top']].set_visible(False)
plt.xlim([-100, 600])

plt.pause(0.0001)

plt.savefig(figure_path + "PhaseModelSyncIPSG.pdf", format = "pdf", bbox_inches = "tight")

