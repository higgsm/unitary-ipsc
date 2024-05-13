# phase_model_async.py
# SIMULATE VTA DOPMAINE NEURON RESPONSES TO ASYNCHRONOUS UNITARY IPSGs.
# Deliver 100 uIPSGs synchronously or in 100, 200, ..., 1000 ms.
# Randomize timing, but keep all uIPSGs at the average amplitude.

import os
import axographio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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

# RESPONSE TO 100 uIPSGs
length = 3
dt = 0.0002
packet_start_time = 0.5
packet_widths = [0.1 * i for i in range(21)]
packet_end_times = [packet_start_time + w for w in packet_widths]
n_ipsgs = 100
n_runs = 500
ipsg_amplitudes = [mean_unitary_peak for i in range(n_ipsgs)]
start_phases = [0.5 / n_runs + i / n_runs for i in range(n_runs)]
n_sets = len(packet_widths)
n_points = round(length / dt)
times = [i * dt for i in range(n_points)]
stimuli = []
model_data = []
random.seed(1)
for s in range(n_sets):
    print('Running set ' + str(s))
    set_stimuli = []
    set_model_data = []
    for r in range(n_runs):
        if packet_widths[s] > 0:
            ipsg_times = [random.uniform(packet_start_time, packet_end_times[s]) for i in range(n_ipsgs)]
            ipsg_times.sort()
        else:
            ipsg_times = [packet_start_time for i in range(n_ipsgs)]
        stim = make_gstim(length, dt, ipsg_times, ipsg_amplitudes, tau_rise, tau_decay)
        set_stimuli.append(stim)
        set_model_data.append(model(stim, dt, start_phases[r], omega))
    stimuli.append(set_stimuli)
    model_data.append(set_model_data)

# RESPONSE TO 50 uIPSGs
length = 3
dt = 0.0002
packet_start_time = 0.5
packet_widths = [0.1 * i for i in range(21)]
packet_end_times = [packet_start_time + w for w in packet_widths]
n_ipsgs = 50
n_runs = 500
ipsg_amplitudes = [mean_unitary_peak for i in range(n_ipsgs)]
start_phases = [0.5 / n_runs + i / n_runs for i in range(n_runs)]
n_sets = len(packet_widths)
n_points = round(length / dt)
times = [i * dt for i in range(n_points)]
stimuli_50 = []
model_data_50 = []
random.seed(1)
for s in range(n_sets):
    print('Running set ' + str(s))
    set_stimuli = []
    set_model_data = []
    for r in range(n_runs):
        if packet_widths[s] > 0:
            ipsg_times = [random.uniform(packet_start_time, packet_end_times[s]) for i in range(n_ipsgs)]
            ipsg_times.sort()
        else:
            ipsg_times = [packet_start_time for i in range(n_ipsgs)]
        stim = make_gstim(length, dt, ipsg_times, ipsg_amplitudes, tau_rise, tau_decay)
        set_stimuli.append(stim)
        set_model_data.append(model(stim, dt, start_phases[r], omega))
    stimuli_50.append(set_stimuli)
    model_data_50.append(set_model_data)

# For comparison, measure average phase delay by one uIPSG (including the spillover into the next ISI)
length = 3
dt = 0.0002
ipsg_time = 0.5
ipsg_amplitude = mean_unitary_peak
n_runs = 500
start_phases = [0.5 / n_runs + i / n_runs for i in range(n_runs)]
n_points = round(length / dt)
times = [i * dt for i in range(n_points)]
stim = make_gstim(length, dt, [ipsg_time], [ipsg_amplitude], tau_rise, tau_decay)
u_model_data = [model(stim, dt, phi0, omega) for phi0 in start_phases]
u_final_phases = [md['phases'][-1] + len(md['spike times']) for md in u_model_data]
final_phases_0 = [phi + omega * length for phi in start_phases]
u_phase_delays = [final_phases_0[r] - u_final_phases[r] for r in range(n_runs)]
mean_u_phase_delay = np.mean(u_phase_delays)
print('mean phase delay for one uIPSG = ' + str(mean_u_phase_delay))

# Mean phase delay, unwrapped, at end of each run
final_phases_0 = [phi + omega * length for phi in start_phases]
final_phases = []
phase_delays = []
phase_delays_means = []
phase_delays_means_50 = []
for s in range(n_sets):
    # 100 uIPSGs
    set_final_phases = [model_data[s][r]['phases'][-1] + len(model_data[s][r]['spike times']) for r in range(n_runs)]
    set_phase_delays = [final_phases_0[r] - set_final_phases[r] for r in range(n_runs)]
    final_phases.append(set_final_phases)
    phase_delays.append(set_phase_delays)
    phase_delays_means.append(np.mean(set_phase_delays))

    # 50 uIPSGs
    set_final_phases = [model_data_50[s][r]['phases'][-1] + len(model_data_50[s][r]['spike times']) for r in range(n_runs)]
    set_phase_delays = [final_phases_0[r] - set_final_phases[r] for r in range(n_runs)]
    phase_delays_means_50.append(np.mean(set_phase_delays))
    

# PUT ALL THE PLOTTING CODE HERE, SO I CAN COPY IT AND RUN IT IN THE TERMINAL!

# FIGURE
print('Making figure ...')
w, h = 7.15, 7.15
plt.ion()
fig = plt.figure(figsize = (w, h))
t_max_plot = 1.8
idx_max = round(t_max_plot / dt)

# Row 1: example stimuli, 100 uIPSGs in 0.1, 0.8 s
s = 1
r1c1 = fig.add_axes([0.08, 0.88, 0.25, 0.08])
sample_times = times[0:idx_max:5]
sample_stimuli = stimuli[s][0][0:idx_max:5]
y_max = max(sample_stimuli) * 1.05
r1c1.plot(sample_times, sample_stimuli, 'k', linewidth = 0.5)
r1c1.spines[['right', 'top', 'bottom']].set_visible(False)
plt.ylim([0, y_max])
plt.xticks([])
plt.title('100 uIPSGs in 100 ms')
plt.ylabel('nS')

s = 8
r1c2 = fig.add_axes([0.39, 0.88, 0.25, 0.08])
r1c2.plot(times[0:idx_max:5], stimuli[s][1][0:idx_max:5], 'k', linewidth = 0.5)
r1c2.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
plt.ylim([0, y_max])
plt.xticks([])
plt.yticks([])
plt.title('100 uIPSGs in 800 ms')

##s = 20
##r1c2 = fig.add_axes([0.74, 0.88, 0.25, 0.08])
##r1c2.plot(times[0:-1:5], stimuli[s][0][0:-1:5], 'k', linewidth = 0.5)
##r1c2.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
##plt.ylim([0, y_max])
##plt.xticks([])
##plt.yticks([])
##plt.title('2 s barrage')

# Row 2: phase trajectories (sample of 20 / 500 (every 25) for 0.1, 0.8, and 2 s barrages)
s = 1
r2c1 = fig.add_axes([0.08, 0.75, 0.25, 0.1])
for r in range(12, n_runs, 25):
    r2c1.plot(times[0:idx_max:5], model_data[s][r]['phases'][0:idx_max:5], 'k', linewidth = 0.5)
r2c1.spines[['right', 'top', 'bottom']].set_visible(False)
plt.xticks([])
plt.ylabel('phase')

s = 8
r2c2 = fig.add_axes([0.39, 0.75, 0.25, 0.1])
for r in range(12, n_runs, 25):
    r2c2.plot(times[0:idx_max:5], model_data[s][r]['phases'][0:idx_max:5], 'k', linewidth = 0.5)
r2c2.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
plt.xticks([])
plt.yticks([])

##s = 20
##r2c3 = fig.add_axes([0.74, 0.75, 0.25, 0.1])
##for r in range(12, n_runs, 25):
##    r2c3.plot(times[0:-1:5], model_data[s][r]['phases'][0:-1:5], 'k', linewidth = 0.5)
##r2c3.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
##plt.xticks([])
##plt.yticks([])

# Row 3: voltage trajectories
s = 1
r3c1 = fig.add_axes([0.08, 0.64, 0.25, 0.08])
for r in range(12, n_runs, 25):
    voltages = [v(phi) for phi in model_data[s][r]['phases']]
    r3c1.plot(times[0:idx_max], voltages[0:idx_max], 'k', linewidth = 0.5)
r3c1.spines[['right', 'top', 'bottom']].set_visible(False)
plt.xticks([])
plt.ylabel('mV')

s = 8
r3c2 = fig.add_axes([0.39, 0.64, 0.25, 0.08])
for r in range(12, n_runs, 25):
    voltages = [v(phi) for phi in model_data[s][r]['phases']]
    r3c2.plot(times[0:idx_max], voltages[0:idx_max], 'k', linewidth = 0.5)
r3c2.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
plt.xticks([])
plt.yticks([])

##s = 20
##r3c3 = fig.add_axes([0.74, 0.64, 0.25, 0.08])
##for r in range(12, n_runs, 25):
##    voltages = [v(phi) for phi in model_data[s][r]['phases']]
##    r3c3.plot(times, voltages, 'k', linewidth = 0.5)
##r3c3.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
##plt.xticks([])
##plt.yticks([])

# Row 4: PSTH
bin_width = 0.01
bins = np.arange(0, t_max_plot + bin_width / 2, bin_width)

s = 1
spike_times = [t for r in range(n_runs) for t in model_data[s][r]['spike times']]
counts, bns = np.histogram(spike_times, bins)
rates = [c / (n_runs * bin_width) for c in counts]
r4c1 = fig.add_axes([0.08, 0.51, 0.25, 0.1])
#r4c1.plot(bins[0:-1], rates, 'k', linewidth = 0.75)
r4c1.fill_between(bins[0:-1], 0, rates, color = 'k')
r4c1.spines[['right', 'top']].set_visible(False)
plt.ylim([0, 26])
plt.xlabel('time (s)')
plt.ylabel('spikes/s')

s = 8
spike_times = [t for r in range(n_runs) for t in model_data[s][r]['spike times']]
counts, bns = np.histogram(spike_times, bins)
rates = [c / (n_runs * bin_width) for c in counts]
r4c2 = fig.add_axes([0.39, 0.51, 0.25, 0.1])
#r4c2.plot(bins[0:-1], rates, 'k', linewidth = 0.75)
r4c2.fill_between(bins[0:-1], 0, rates, color = 'k')
r4c2.spines[['right', 'top']].set_visible(False)
plt.ylim([0, 26])
plt.xlabel('time (s)')

##s = 20
##spike_times = [t for r in range(n_runs) for t in model_data[s][r]['spike times']]
##counts, bns = np.histogram(spike_times, bins)
##rates = [c / (n_runs * bin_width) for c in counts]
##r4c3 = fig.add_axes([0.74, 0.53, 0.25, 0.08])
##r4c3.plot(bins[0:-1], rates, 'k', linewidth = 0.75)
##r4c3.spines[['right', 'top']].set_visible(False)
##plt.xlabel('time (s)')

# Row 5: mean phase delay vs. packet width
r5c1 = fig.add_axes([0.74, 0.61, 0.25, 0.25])
r5c1.plot(packet_widths, phase_delays_means, 'k', marker = '.', linestyle = 'none')
r5c1.plot(packet_widths, phase_delays_means_50, 'gray', marker = '.', linestyle = 'none')
r5c1.axhline(100 * mean_u_phase_delay, color = 'black', linestyle = '--')
r5c1.axhline(50 * mean_u_phase_delay, color = 'gray', linestyle = '--')
plt.ylim([0, 1.05 * max(phase_delays_means)])
r5c1.spines[['right', 'top']].set_visible(False)
plt.xlabel('barrage duration (s)')
plt.ylabel('mean phase delay')

plt.pause(0.0001)
plt.savefig(figure_path + "PhaseModelAsync.pdf", format = "pdf", bbox_inches = "tight")

