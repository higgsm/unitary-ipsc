# mh_vclamp.py
# DEFINE ANALYSIS FUNCTIONS FOR VOLTAGE CLAMP DATA

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

# load voltage clamp data from Axograph file
def load_vclamp(data_path):

    # list Axograph files in specified directory
    files = []
    count = 0
    for fname in sorted(os.listdir(data_path)):
        if fname.endswith(".axgd"):
            file = fname[0:-5]
            files.append(file)
            print(str(count) + ', ' + file)
            count += 1
    file_paths = [data_path + file for file in files]

    # user enters a file number
    file_number = int(input('Choose file number: '))

    # load data from the file
    file = files[file_number]
    axgd = axographio.read(data_path + file + '.axgd')
    times = np.asarray(axgd.data[0])
    currents = np.asarray(axgd.data[1:]) * 1e12
    dt = times[1] - times[0]
    if (dt > 0.000019) and (dt < 0.000021):
        dt = 0.00002
    if (dt > 0.000049) and (dt < 0.000051):
        dt = 0.00005
    [n_episodes, n_points] = currents.shape
    episodes = list(range(n_episodes))
    episode_length = n_points * dt

    # return the data as a dictionary
    return {
        'filename': file,
        'times': times,
        'currents': currents,
        'dt': dt,
        'n episodes': n_episodes,
        'n points': n_points,
        'episodes': episodes,
        'episode length': episode_length
        }

# plot voltage clamp data (raw)
def plot_vclamp(data, plot_episodes, t_min, t_max, y_min, y_max, figure_path):
    idx_min = round(t_min / data['dt'])
    idx_max = round(t_max / data['dt'])
    for episode in plot_episodes:
        x_values = data['times'][idx_min:idx_max]
        y_values = data['currents'][episode][idx_min:idx_max]
        plt.plot(x_values, y_values, 'k', linewidth = 0.75)
    if y_max > y_min:
        plt.ylim([y_min, y_max])
    plt.xlabel('time (ms)', fontsize = 12)
    plt.ylabel('current (pA)', fontsize = 12)
    plt.tight_layout()
    if figure_path != 'none':
        plt.savefig(figure_path + "myImage.pdf", format="pdf", bbox_inches="tight")
    plt.show()

# measure mean current and access resistance for each episode; each episode must include a voltage pulse
def measure_access(data, pulse_amplitude, pulse_time):
    currents = data['currents']
    dt = data['dt']
    pulse_point = round(pulse_time / dt)
    sample_width = 0.0005
    mean_currents = np.average(currents, axis = 1)
    sample_points = round(sample_width / dt)
    access_resistances = []
    for episode in data['episodes']:
        baseline = mean(currents[episode, pulse_point - sample_points: pulse_point])
        if pulse_amplitude > 0:
            maxval = np.max(currents[episode, pulse_point + 1: pulse_point + sample_points])
        else:
            maxval = np.min(currents[episode, pulse_point + 1: pulse_point + sample_points])
        peak_amp = maxval - baseline
        access_resistance = pulse_amplitude / (peak_amp * 1e-6)
        access_resistances.append(access_resistance)
    
    #out_path = os.getcwd() + '/Data/Access/' + file + '_AccessData.csv'
    #with open(out_path, "w") as f:
    #    wr = csv.writer(f)
    #    wr.writerow(['episode', 'mean current (pA)','access resistance (MOhm)'])
    #    for episode in episodes:
    #        wr.writerow([episode, mean_currents[episode], access_resistances[episode]])
    
    plt.plot(mean_currents,'k', marker = '.', markersize = 10, linestyle = 'none')
    plt.grid()
    plt.xlabel('episode', fontsize = 14)
    plt.ylabel('mean current (pA)', fontsize = 14)
    plt.show()

    plt.plot(access_resistances,'k', marker = '.', markersize = 10, linestyle = 'none')
    plt.ylim(0, 1.1 * max(access_resistances))
    plt.grid()
    plt.xlabel('episode', fontsize = 14)
    plt.ylabel('access resistance (MOhm)', fontsize = 14)
    plt.show()
    
    return {
        'mean currents (pA)': mean_currents,
        'access resistances (MOhm)': access_resistances
    }

# Subtract baseline, taken as the median value in a time window, from each current trace
# THIS SHOULD OPERATE ON THE DATA PASSED TO THE FUNCTION.
def baseline_subtract(data, t_min, t_max):
    idx_min = round(t_min / data['dt'])
    idx_max = round(t_max / data['dt'])
    baselines = []
    for episode in data['episodes']:
        base = median(data['currents'][episode][idx_min: idx_max])
        data['currents'][episode] -= base
        
# Find PSCs in one episode
def find_pscs(current, dt, t_min, t_max):
    
    # smooth with Gaussian filter (SD = 0.2 ms), take derivative, normalize by median absolute value
    sigma = 0.0002
    filter_times = np.arange(-5 * sigma, 5 * sigma, dt)
    filter_values = np.exp(-filter_times**2 / (2 * sigma**2))
    filter_values /= filter_values.sum()
    padded_current = np.append(current, np.full(len(filter_values), current[-1]))
    smoothed_current = np.convolve(padded_current, filter_values)
    start_point = round(len(filter_values) / 2)
    smoothed_current = smoothed_current[start_point:start_point + len(current)]
    diffs = np.diff(smoothed_current)
    diffs /= np.median(np.abs(diffs))
    
    # find PSC times at minimum of (smoothed) derivative
    detection_level = -6
    peak_measure_start = 0
    peak_measure_end = 0.002
    idx_min = round(t_min / dt)
    idx_max = round(t_max / dt)
    peak_start_offset = round(peak_measure_start / dt)
    peak_end_offset = round(peak_measure_end / dt)
    psc_times = []
    psc_amplitudes = []
    for i in range(idx_min, idx_max):
        d0 = diffs[i-1]
        d1 = diffs[i]
        d2 = diffs[i+1]
        if d1 < d0 and d1 <= d2 and d1 < detection_level:
                peak = min(smoothed_current[i + peak_start_offset: i + peak_end_offset])
                psc_times.append(i * dt)
                psc_amplitudes.append(peak)
    
    # return PSC times and amplitudes
    return {
        'n PSCs': len(psc_times),
        'PSC times (s)': psc_times,
        'PSC amplitudes (pA)': psc_amplitudes,
    }

# find PSCs in all episodes (use global currents)
def find_all_pscs(data, t_min, t_max):
    currents = data['currents']
    episodes = data['episodes']
    dt = data['dt']
    n_pscs = []
    psc_times = []
    psc_amplitudes = []
    for episode in episodes:
        psc_data = find_pscs(currents[episode], dt, t_min, t_max)
        n_pscs.append(psc_data['n PSCs'])
        psc_times.append(psc_data['PSC times (s)'])
        psc_amplitudes.append(psc_data['PSC amplitudes (pA)'])
        
    # colect data for each PSC (episode, time, amplitude) from all episodes into a single list
    all_psc_times = flatten(psc_times)
    all_psc_amplitudes = flatten(psc_amplitudes)
    all_psc_episodes = []
    for episode in episodes:
        for i in range(n_pscs[episode]):
            all_psc_episodes.append(episode)
        
    # write PSC data to file
    #with open(output_data_path, "w") as f:
    #    wr = csv.writer(f)
    #    wr.writerow(['episode', 'IPSC time (s)','IPSC amplitude (pA)'])
    #    for i in range(len(all_psc_times)):
    #        wr.writerow([all_psc_episodes[i], all_psc_times[i], all_psc_amplitudes[i]])
    #print('PSC data saved to ' + output_data_path)
    
    # return results; each item is a list with a subitem/sublist for each episode
    return {
        'n PSCs': n_pscs,
        'PSC times (s)': psc_times,
        'PSC amplitudes (pA)': psc_amplitudes,
    }

# fit synaptic current waveform with double-exponential decay
# ELIMINATE THE EXTRA SINGLE-EXPONENTIAL FIT FOR CASES WHERE THE TWO DECAY RATES ARE NOT DIFFERENT.
# INSTEAD, RETURN ERROR FLAGS IN THAT CASE.
def fit_psc(psc_times, psc, baseline_start, t0_guess, a_fast_guess, a_slow_guess, k_rise_guess, k_d1_guess, k_d2_guess):
    
    dt = psc_times[1] - psc_times[0]

    # define fitting function: exponential rise, double-exponential decay
    def psc_function(t, t0, a1, a2, kr, kd1, kd2):
        y = -(a1 + a2) * np.exp(-kr * (t - t0)) + a1 * np.exp(-kd1 * (t - t0)) + a2 * np.exp(-kd2 * (t-t0))
        return y

    # list of parameter guesses
    guess = [t0_guess, a_fast_guess, a_slow_guess, k_rise_guess, k_d1_guess, k_d2_guess]

    # data to fit
    fit_start_point = round(-baseline_start / dt)
    fit_times = psc_times[fit_start_point:]
    fit_currents = psc[fit_start_point:]

    # do the fit
    parameters, covariance = curve_fit(psc_function, fit_times, fit_currents, p0 = guess)

    # collect the fit parameters
    fit_t0 = parameters[0]
    fit_a1 = parameters[1]
    fit_a2 = parameters[2]
    fit_kr = parameters[3]
    fit_kd1 = parameters[4]
    fit_kd2 = parameters[5]

    # make sure the two decay rates are different, then collect results
    fit_kd_ratio = fit_kd1 / fit_kd2
    if fit_kd_ratio < 1:
        fit_kd_ratio = fit_kd2 / fit_kd1
    if (fit_kd2 > 0) and (fit_a1 < 0) and (fit_a2 < 0) and (fit_kd1 > 0) and (fit_kd2 > 0) and (fit_kd_ratio > 1.01):
        tau_r = 1000 / fit_kr
        if fit_kd2 < fit_kd1:
            a_f = fit_a1
            a_s = fit_a2
            k_f = fit_kd1
            k_s = fit_kd2
        else:
            a_f = fit_a2
            a_s = fit_a1
            k_f = fit_kd2
            k_s = fit_kd1
        tau_f = 1000 / k_f
        tau_s = 1000 / k_s
        fraction_fast = a_f / (a_f + a_s)
        tau_w = (a_f * tau_f + a_s * tau_s) / (a_f + a_s)
        fit_values = psc_function(fit_times, fit_t0, a_f, a_s, fit_kr, k_f, k_s)
    else:
        tau_r = 'ERROR'
        tau_f = 'ERROR'
        tau_s = 'ERROR'
        fraction_fast = 'ERROR'
        tau_w = 'ERROR'
        fit_times = 'ERROR'
        fit_values = 'ERROR'
        
    results = {
        'tau(r)': tau_r,
        'tau(f)': tau_f,
        'tau(s)': tau_s,
        'fraction fast': fraction_fast,
        'tau(w)': tau_w,
        'fit times': fit_times,
        'fit values': fit_values
        }

    # return results
    return results

# fit synaptic current waveform with single-exponential decay
def fit_psc_single(psc_times, psc, baseline_start, t0_guess, a_guess, k_rise_guess, k_d_guess):
    
    dt = psc_times[1] - psc_times[0]

    # define fitting function: exponential rise, double-exponential decay
    def psc_function(t, t0, a, kr, kd):
        y = -a * np.exp(-kr * (t - t0)) + a * np.exp(-kd * (t - t0))
        return y

    # list of parameter guesses
    guess = [t0_guess, a_guess, k_rise_guess, k_d_guess]

    # data to fit
    fit_start_point = round(-baseline_start / dt)
    fit_times = psc_times[fit_start_point:]
    fit_currents = psc[fit_start_point:]

    # do the fit
    parameters, covariance = curve_fit(psc_function, fit_times, fit_currents, p0 = guess)

    # collect the fit parameters
    fit_t0 = parameters[0]
    fit_a = parameters[1]
    fit_kr = parameters[2]
    fit_kd = parameters[3]
    tau_r = 1000 / fit_kr
    tau_d = 1000 / fit_kd
    fit_values = psc_function(psc_times[fit_start_point:], fit_t0, fit_a, fit_kr, fit_kd)
    results = {
        'tau(r)': tau_r,
        'tau(d)': tau_d,
        'fit times': fit_times,
        'fit values': fit_values
        }

    # return results
    return results

