# phase_model_barrage.py
# SIMULATE VTA DOPMAINE NEURON RESPONSES TO BARRAGES OF UNITARY IPSGs

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

uipsc_means = [
    -37.1,
    -51.3,
    -122.7,
    -38.0,
    -113.1,
    -94.4,
    -87.3,
    -112.3,
    -32.1,
    -151.1,
    -63.7,
    -42.6,
    -111.3,
    -75.1,
    -99.9,
    -77.0,
    -65.2,
    -86.5,
    -107.2,
    -75.8,
    -34.7,
    -719.1,
    -138.8,
    -105.0,
    -113.6,
    -96.0,
    -221.5,
    -287.7,
    -62.0,
    -69.2
    ]
uipsc_sds = [
    4.4,
    11.1,
    23.6,
    12.3,
    34.5,
    23.5,
    13.3,
    37.7,
    10.1,
    84.6,
    24.9,
    24.2,
    20.6,
    30.1,
    39.4,
    34.9,
    11.6,
    64.1,
    56.8,
    33.0,
    9.8,
    296.8,
    72.8,
    25.8,
    57.9,
    82.8,
    44.1,
    185.2,
    39.6,
    23.2
    ]
n_afferents = len(uipsc_means)
uipsg_means = [mu / -67 for mu in uipsc_means]
uipsg_sds = [sigma / 67 for sigma in uipsc_sds]

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

##plot_phases = np.arange(0, 1, 0.001)
##plot_zs = [z(phi) for phi in plot_phases]
##plt.ioff()
##plt.plot(plot_phases, plot_zs, 'k')
##plt.show()

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

##traj_phases = [t / traj_times[-1] for t in traj_times]
##plt.plot(traj_phases, traj_voltages, 'k')
##plt.show()

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

# RESPONSE TO LONG BOMBARDMENT AT A RANGE OF uIPSG RATES
uipsg_rates = [1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
n_runs = len(uipsg_rates)
length = 100
dt = 0.0001
n_points = round(length / dt)
times = [i * dt for i in range(n_points)]
phi0 = 0
stimuli = []
model_data = []
random.seed(1)
for r in range(n_runs):
    print('run ' + str(r) + ' ...')
    n_ipsgs = round(length * uipsg_rates[r])
    ipsg_times = [random.uniform(0, length) for i in range(n_ipsgs)]
    ipsg_times.sort()
    afferents = [random.randint(0, n_afferents - 1) for i in range(n_ipsgs)]
    ipsg_amplitudes = []
    for a in afferents:
        mu = uipsg_means[a]
        sigma = uipsg_sds[a]
        alpha = mu * mu / (sigma * sigma)
        beta = (sigma * sigma) / mu
        ipsg_amplitudes.append(np.random.gamma(alpha, beta))
    stim = make_gstim(length, dt, ipsg_times, ipsg_amplitudes, tau_rise, tau_decay)
    stimuli.append(stim)
    model_data.append(model(stim, dt, phi0, omega))

# Spike rates, CV of ISIs
phases = [md['phases'] for md in model_data]
spike_times = [md['spike times'] for md in model_data]
spike_counts = [len(ts) for ts in spike_times]
spike_rates = [c / length for c in spike_counts]
isis = [np.diff(ts) for ts in spike_times]
cvs = []
r = 0
while (len(isis[r]) >= 10) and (r < n_runs):
    cvs.append(np.std(isis[r]) / np.mean(isis[r]))
    r += 1

# Voltage across each run
print('Collecting voltage trajectories ...')
voltages = []
for phis in phases:
    voltages.append([v(phi) for phi in phis])

# Sensitivity across each run
print('Calculating time-varying sensitivity ...')
sens = []
for phis in phases:
    sens.append([-z(phi) * (e_rev - v(phi)) for phi in phis])
mean_sens = [np.mean(ss) for ss in sens]

# FIGURE
print('Making figure ...')
w, h = 7.15, 7.15
plt.ion()
fig = plt.figure(figsize = (w, h))

# Row 1, left: uIPSG barrages
r = 6
ax1a = fig.add_axes([0.1, 0.88, 0.25, 0.08])
ax1a.plot(times[:100000:10], stimuli[r][:100000:10], 'k', linewidth = 0.75)
ax1a.spines[['right', 'top', 'bottom']].set_visible(False)
plt.xticks([])
plt.title(str(uipsg_rates[r]) + ' uIPSGs/s')
plt.ylabel('nS')

# Row 1, right
r = 14
ax1b = fig.add_axes([0.6, 0.88, 0.25, 0.08])
ax1b.plot(times[:100000:10], stimuli[r][:100000:10], 'k', linewidth = 0.75)
ax1b.spines[['right', 'top', 'bottom']].set_visible(False)
plt.title(str(uipsg_rates[r]) + ' uIPSGs/s')
plt.xticks([])

# Row 2, left: phase trajectories and densities
r = 6
ax2a = fig.add_axes([0.1, 0.7, 0.25, 0.15])
ax2a.plot(times[:100000:10], phases[r][:100000:10], 'k', linewidth = 0.75)
ax2a.spines[['right', 'top', 'bottom']].set_visible(False)
plt.xticks([])
plt.ylabel('phase')

ax2b = fig.add_axes([0.4, 0.7, 0.08, 0.15])
bns = np.arange(0, 1, 0.01)
ax2b.hist(phases[r], bins = bns, orientation = 'horizontal', density = 'True', color = 'k')
ax2b.spines[['left', 'right', 'top']].set_visible(False)
plt.yticks([])
plt.xlabel('density')

# Row 2, right
r = 14
ax2c = fig.add_axes([0.6, 0.7, 0.25, 0.15])
ax2c.plot(times[:100000:10], phases[r][:100000:10], 'k', linewidth = 0.75)
ax2c.spines[['right', 'top', 'bottom']].set_visible(False)
plt.xticks([])

ax2d = fig.add_axes([0.9, 0.7, 0.08, 0.15])
bns = np.arange(0, 1, 0.01)
ax2d.hist(phases[r], bins = bns, orientation = 'horizontal', density = 'True', color = 'k')
ax2d.spines[['left', 'right', 'top']].set_visible(False)
ax2d.spines[['left', 'right', 'top']].set_visible(False)
plt.yticks([])
plt.xlabel('density')

# Row 3, left: V(phi) trajectories
r = 6
ax3a = fig.add_axes([0.1, 0.59, 0.25, 0.08])
ax3a.plot(times[:100000:5], voltages[r][:100000:5], 'k', linewidth = 0.75)
ax3a.spines[['right', 'top']].set_visible(False)
plt.xlabel('time (s)')
plt.ylabel('mV')

# Row 3, right
r = 14
ax3b = fig.add_axes([0.6, 0.59, 0.25, 0.08])
ax3b.plot(times[:100000:5], voltages[r][:100000:5], 'k', linewidth = 0.75)
ax3b.spines[['right', 'top']].set_visible(False)
plt.xlabel('time (s)')

# Row 4, left: firing rate vs. uIPSG rate
ax4a = fig.add_axes([0.1, 0.35, 0.2, 0.15])
ax4a.plot(uipsg_rates, spike_rates, 'k', marker = '.', linestyle = 'none')
ax4a.spines[['right', 'top']].set_visible(False)
plt.xlabel('uIPSG rate (/s)')
plt.ylabel('spikes/s')

# Row 4, middle: CV vs. uIPSG rate
ax4b = fig.add_axes([0.43, 0.35, 0.2, 0.15])
ax4b.plot(uipsg_rates[:len(cvs)], cvs, 'k', marker = '.', linestyle = 'none')
ax4b.spines[['right', 'top']].set_visible(False)
plt.yticks([0, 0.25, 0.5, 0.75])
plt.xlabel('uIPSG rate (/s)')
plt.ylabel('CV of ISIs')

# Row 4, right: mean sensitivity vs. uIPSG rate
ax4b = fig.add_axes([0.76, 0.35, 0.2, 0.15])
ax4b.plot(uipsg_rates, mean_sens, 'k', marker = '.', linestyle = 'none')
plt.ylim([0, 1.1 * max(mean_sens)])
ax4b.spines[['right', 'top']].set_visible(False)
plt.yticks([0, 0.25, 0.5, 0.75])
plt.xlabel('uIPSG rate (/s)')
plt.ylabel('mean sensitivity\n(cycles/nS-s)')

plt.pause(0.0001)
plt.savefig(figure_path + "PhaseModelBarrage.pdf", format = "pdf", bbox_inches = "tight")
