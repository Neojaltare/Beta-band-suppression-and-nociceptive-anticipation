
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
from scipy.stats import linregress
from scipy.signal import find_peaks
import nitime.analysis as nta
import nitime.timeseries as ts
import numpy as np
import mne
from mne.time_frequency import psd_array_multitaper
import os
import re




# Define functions for correction
def infer_rights(responses, srate, left_idx):
    pportstarted = responses['p_port_LAS.started'][responses['p_port'].isin([3, 5, 8])] + responses['stimulation_LAS.started'][responses['p_port'].isin([3, 5, 8])]
    p_port = responses['p_port'][responses['p_port'].isin([3, 5, 8])]

    for idx, marker in enumerate(p_port):
        if marker == 3:
            firstleft = idx
            firstleftsec = pportstarted.values[firstleft]
            break

    rightseconds = []
    for idx, marker in enumerate(p_port):
        if marker == 5:
            secondsdiff = pportstarted.values[idx] - firstleftsec
            rightseconds.append(secondsdiff*srate)

    rightsamples = np.array(rightseconds)
    right_idx = left_idx[0] + rightsamples
    return np.round(right_idx).astype(int)


# p_port: 5 = Right, 3 = Left, 8 = Invalid
# Function to infer right-side markers and apply drift correction
def correct_drift(responses, srate, left_idx, right_idx):
    # Time in psychopy for left markers (in seconds and samples)
    pportstarted_seconds_left = (responses['p_port_LAS.started'][responses['p_port'].isin([3])] 
                                 + responses['stimulation_LAS.started'][responses['p_port'].isin([3])])
    pportstarted_seconds_left = pportstarted_seconds_left.values - pportstarted_seconds_left.values[0]

    # Create psychopy time vector for left markers
    psychopy_timevec_left = np.linspace(pportstarted_seconds_left[0], pportstarted_seconds_left[-1], 
                                        np.round((pportstarted_seconds_left[-1] - pportstarted_seconds_left[0]) * srate).astype(int))

    indices = np.clip(np.round(pportstarted_seconds_left * srate).astype(int), 0, len(psychopy_timevec_left) - 1)
    psychopy_left_stims = psychopy_timevec_left[indices]

    # Left stimulations in the data
    left_idx_seconds = left_idx / srate
    left_idx_seconds -= left_idx_seconds[0]  # Normalize
    data_timevec_left = np.linspace(left_idx_seconds[0], left_idx_seconds[-1], 
                                    np.round((left_idx_seconds[-1] - left_idx_seconds[0]) * srate).astype(int))

    indices = np.clip(np.round(left_idx_seconds * srate).astype(int), 0, len(data_timevec_left) - 1)
    data_left_stims = data_timevec_left[indices]

    # Calculate differences between the two stim times (left side)
    differences_left = psychopy_left_stims - data_left_stims

    # Calculate slope per unit time (samples per second)
    slope = (differences_left[-1] - differences_left[0]) / (data_timevec_left[-1] - data_timevec_left[0])
    print(f'Slope in samples per second: {slope}')

    # Correct the right-side markers
    pportstarted_seconds_right = (responses['p_port_LAS.started'][responses['p_port'].isin([5])] 
                                  + responses['stimulation_LAS.started'][responses['p_port'].isin([5])])
    psychopy_first = pportstarted_seconds_right.values[0]
    pportstarted_seconds_right = pportstarted_seconds_right.values - pportstarted_seconds_right.values[0]

    psychopy_timevec_right = np.linspace(pportstarted_seconds_right[0], pportstarted_seconds_right[-1], 
                                         np.round((pportstarted_seconds_right[-1] - pportstarted_seconds_right[0]) * srate).astype(int))
    
    indices = np.clip(np.round(pportstarted_seconds_right * srate).astype(int), 0, len(psychopy_timevec_right) - 1)
    psychopy_right_stims = psychopy_timevec_right[indices]

    data_first = right_idx[0] / srate
    Overlap = np.max([data_first, psychopy_first]) - np.min([data_first, psychopy_first])
    print('Overlap:', Overlap)

    right_idx_seconds = right_idx / srate
    assert len(right_idx_seconds) == len(pportstarted_seconds_right), 'Lengths of right stimulations and psychopy stimulations do not match'

    # Apply correction to right markers
    corrected_right_idx = np.zeros(len(right_idx))
    for idx, seconds in enumerate(pportstarted_seconds_right):
        if idx == 0:
            correction = Overlap * slope
        else:
            correction = (right_idx[idx] - right_idx[0]) * slope - (Overlap * slope)
        print(f'Idx: {idx}, Correction: {correction}')
        corrected_right_idx[idx] = right_idx[idx] - correction

    corrected_right_idx = np.round(corrected_right_idx).astype(int)

    # Create time vector for the corrected right indices
    right_idx_seconds = corrected_right_idx / srate
    right_idx_seconds -= right_idx_seconds[0]  # Normalize
    data_timevec_right = np.linspace(right_idx_seconds[0], right_idx_seconds[-1], 
                                     np.round((right_idx_seconds[-1] - right_idx_seconds[0]) * srate).astype(int))

    indices = np.clip(np.round(right_idx_seconds * srate).astype(int), 0, len(data_timevec_right) - 1)
    data_right_stims = data_timevec_right[indices]

    # Calculate corrected differences for right stimulations
    corrected_differences_right = psychopy_right_stims - data_right_stims

    # Plot results
    plt.plot(differences_left, label='Left')
    plt.plot(corrected_differences_right, label='Corrected Right')
    plt.legend()
    plt.title('Drift Corrected Stimulation Differences')
    plt.show()

    return corrected_right_idx


def find_correlation_peaks(raw_data,norm_template_28):
    """
    Finds peaks in the cross-correlation between a raw data signal and a template.

    Parameters:
    ----------
    raw_data : numpy.ndarray
        2D array of shape (channels, samples), containing the raw signals for multiple channels.
    template_28 : numpy.ndarray
        1D or 2D array representing the template to match against the raw data.

    Returns:
    -------
    All_peaks : list of numpy.ndarray
        A list where each element corresponds to the indices of detected peaks for a channel.
    """

    # Define your template and scaling factors
    correlations = np.zeros((raw_data.shape[0], raw_data.shape[1]))

    for channel in range(raw_data.shape[0]):
        correlations[channel, :] = np.correlate(raw_data[channel, :], norm_template_28, mode='same')

    corr_mean = np.zeros((raw_data.shape[0], raw_data.shape[1]))
    corr_std = np.zeros((raw_data.shape[0], raw_data.shape[1]))
    All_peaks = []
    for channel in range(raw_data.shape[0]):    
        window_size = 4000
        corr_mean[channel, :] = np.convolve(correlations[channel, :], np.ones(window_size)/window_size, mode='same')
        corr_std[channel, :] = np.sqrt(np.convolve(correlations[channel, :]**2, np.ones(window_size)/window_size, mode='same') - corr_mean[channel, :]**2)
        
        threshold = corr_mean[channel, :] + corr_std[channel, :]
        peaks, _ = find_peaks(correlations[channel, :], height=threshold)
        All_peaks.append(peaks)
    return All_peaks


def extract_correlations(raw_data,norm_template_28,All_peaks):
    # Loop through all peaks to gather the correlations
    All_correlations = {'left': np.zeros(len(All_peaks[0])), 'right': np.zeros(len(All_peaks[1]))}
    template_length = len(norm_template_28)
    for channel in range(raw_data.shape[0]):
        for idx, peak in enumerate(All_peaks[channel]):
            tempsegment = raw_data[channel,peak-(template_length//2):peak+(template_length//2)]
            if len(tempsegment) < template_length:
                continue
            correlation = np.corrcoef(tempsegment, norm_template_28)[0, 1]
            if channel == 0:
                All_correlations['left'][idx] = correlation
            else:
                All_correlations['right'][idx] = correlation
    assert All_correlations['left'].shape[0] == len(All_peaks[0]), 'Correlation shape mismatch'
    assert All_correlations['right'].shape[0] == len(All_peaks[1]), 'Correlation shape mismatch'
    return All_correlations


def correct_data_with_template(raw_data, norm_template_28, All_peaks, corrthresh=0.8):
    """
    Correct data segments based on a template match.

    Parameters:
        raw_data (np.ndarray): The raw data array (channels x samples).
        norm_template_28 (np.ndarray): The normalized template to match.
        All_peaks (list of lists): Indices of detected peaks for each channel.
        corrthresh (float): Threshold for correlation to consider a match.

    Returns:
        Corrected_data (np.ndarray): The corrected data.
        All_errors (dict): Squared error for left and right channels.
    """

    template_length = len(norm_template_28)
    Corrected_data = np.copy(raw_data)

    # Taper the edges of the template
    negative_ramp = [0, 50]
    positive_ramp = [150, template_length]
    ramp_up = np.hanning((negative_ramp[1] - negative_ramp[0]) * 2)[:(negative_ramp[1] - negative_ramp[0])]
    ramp_down = np.hanning((positive_ramp[1] - positive_ramp[0]) * 2)[(positive_ramp[1] - positive_ramp[0]):]
    tapered_template = np.copy(norm_template_28)
    print(tapered_template.shape)
    tapered_template[:negative_ramp[1]] *= ramp_up
    tapered_template[positive_ramp[0]:] *= ramp_down

    # Define the windows for scaling
    negative_win = [45, 80]
    positive_win = [115, 150]

    # Initialize error dictionary
    All_errors = {'left': np.zeros(len(All_peaks[0])), 'right': np.zeros(len(All_peaks[1]))}

    # Process each channel and peak
    for channel in range(Corrected_data.shape[0]):
        for idx, peak in enumerate(All_peaks[channel]):
            tempsegment = Corrected_data[channel, peak - (template_length // 2):peak + (template_length // 2)]
            if len(tempsegment) != template_length:
                continue

            # Check if it is a match
            correlation = np.corrcoef(tempsegment, norm_template_28)[0, 1]
            if correlation > corrthresh:
                scaling_window = [negative_win[0], positive_win[1]]
                slope, intercept, _, _, _ = linregress(
                    tapered_template[scaling_window[0]:scaling_window[1]],
                    tempsegment[scaling_window[0]:scaling_window[1]]
                )
                scaled_template = tapered_template * slope + intercept

                # Further scale each peak individually
                scaled_template[positive_win[0]:positive_win[1]] *= (
                    tempsegment[positive_win[0]:positive_win[1]].max() / scaled_template.max()
                )
                scaled_template[negative_win[0]:negative_win[1]] *= (
                    tempsegment[negative_win[0]:negative_win[1]].min() / scaled_template.min()
                )

                # Subtract the scaled template
                tempsegment = tempsegment - scaled_template

                # Calculate squared error
                error = np.sum((tempsegment - norm_template_28))
                if channel == 0:
                    All_errors['left'][idx] = error
                elif channel == 1:
                    All_errors['right'][idx] = error

                # Update the corrected data
                Corrected_data[channel, peak - (template_length // 2):peak + (template_length // 2)] = tempsegment

    return Corrected_data, All_errors


def correct_data_with_template_2(Input_data, template_2, All_peaks, corrthresh=0.8, smooth = False):
    """
    Correct data segments based on a template match.

    Parameters:
        Input_data (np.ndarray): The raw data array (channels x samples).
        norm_template_2 (np.ndarray): The normalized template to match.
        All_peaks (list of lists): Indices of detected peaks for each channel.
        corrthresh (float): Threshold for correlation to consider a match.

    Returns:
        Corrected_data_2 (np.ndarray): The corrected data.
        All_errors (dict): Squared error for left and right channels.
    """
    template_length = len(template_2)
    Corrected_data_2 = np.copy(Input_data)
    if smooth:
        from scipy.ndimage import gaussian_filter1d
        template_2 = gaussian_filter1d(template_2, sigma=6)
    # Taper the edges of the template
    negative_ramp = [0, 100]
    positive_ramp = [template_length - 100, template_length]
    ramp_up = np.hanning((negative_ramp[1] - negative_ramp[0]) * 2)[:(negative_ramp[1] - negative_ramp[0])]
    ramp_down = np.hanning((positive_ramp[1] - positive_ramp[0]) * 2)[(positive_ramp[1] - positive_ramp[0]):]
    tapered_template = np.copy(template_2)
    tapered_template[:negative_ramp[1]] *= ramp_up
    tapered_template[positive_ramp[0]:] *= ramp_down
    subtracted_segments = []
    # Define the windows for scaling
    negative_win = [187, 217] 
    positive_win = [120, 150]

    # Initialize error dictionary
    All_errors = {'left': np.zeros(len(All_peaks[0])), 'right': np.zeros(len(All_peaks[1]))}
    subtracted_segments = {
    'left': np.zeros((len(All_peaks[0]), template_length)), 
    'right': np.zeros((len(All_peaks[1]), template_length))}
    # Process each channel and peak
    for channel in range(Corrected_data_2.shape[0]):
        for idx, peak in enumerate(All_peaks[channel]):
            tempsegment = Corrected_data_2[channel, peak - (template_length // 2):peak + (template_length // 2)]
            if len(tempsegment) != template_length:
                continue

            # Check if it is a match
            correlation = np.corrcoef(tempsegment, template_2)[0, 1]
            if correlation > corrthresh:
                scaling_window = [positive_win[0], negative_win[1]]
                slope, intercept, _, _, _ = linregress(
                    tapered_template[scaling_window[0]:scaling_window[1]],
                    tempsegment[scaling_window[0]:scaling_window[1]]
                )
                scaled_template = tapered_template * slope + intercept

                # Further scale each peak individually
                scaled_template[positive_win[0]:positive_win[1]] *= (
                    tempsegment[positive_win[0]:positive_win[1]].max() / scaled_template.max()
                )
                scaled_template[negative_win[0]:negative_win[1]] *= (
                    tempsegment[negative_win[0]:negative_win[1]].min() / scaled_template.min()
                )

                # Subtract the scaled template
                subtracted_seg = tempsegment - scaled_template

                # Calculate squared error
                squared_residuals = (subtracted_seg) ** 2
                error = np.sum(squared_residuals)
                if channel == 0:
                    All_errors['left'][idx] = error
                    subtracted_segments['left'][idx] = subtracted_seg
                elif channel == 1:
                    All_errors['right'][idx] = error
                    subtracted_segments['right'][idx] = subtracted_seg

                # Update the corrected data
                Corrected_data_2[channel, peak - (template_length // 2):peak + (template_length // 2)] = subtracted_seg

    return Corrected_data_2, All_errors, subtracted_segments

# Function to check that the trials order matches in the EEG and EMG data
def Check_trial_match(Subject_EEG, Subject_EMG):
    EEG_events = Subject_EEG.events
    EMG_events = Subject_EMG.events

    EEG_events_copy = EEG_events.copy()
    # Match the events between the EEG and EMG data
    EEG_events_copy[:,2][EEG_events_copy[:,2] == 10002] = 1
    EEG_events_copy[:,2][EEG_events_copy[:,2] == 10004] = 2
    EEG_events_copy[:,2][EEG_events_copy[:,2] == 10003] = 3

    return np.all(EEG_events_copy[:,2] - EMG_events[:,2] == 0)

# Load the EEG and corrosponding EMG data
def Load_EEG_EMG_Paths():
    EEG_path = "/Volumes/KetanData2/BBO/EEG/Processed/"
    EMG_path = "/Volumes/KetanData2/BBO/EMG/Processed/"
    EMG_filenames = []
    EEG_filenames = []
    Behavioural_filenames = []
    EMG_folders = [f for f in os.listdir(EMG_path) if os.path.isdir(os.path.join(EMG_path, f))]
    EEG_folders = [f for f in os.listdir(EEG_path) if os.path.isdir(os.path.join(EEG_path, f))]
    for idx, folder in enumerate(EEG_folders):
        if 'BBO' not in folder:
            continue
        participant_id = int(folder.split('_')[1])

        # Find the corrosponding EMG folder
        for EMG_idx, EMG_folder in enumerate(EMG_folders):
            match = re.search(r'\d+', EMG_folder)
            if match:
                folder_number = int(match.group())  # Convert the extracted number to an integer
                if folder_number == participant_id:
                    EMG_to_extract = EMG_folder
                    break
        EMG_file = [f for f in os.listdir(os.path.join(EMG_path, EMG_to_extract)) if 'EMG-epo' in f]
        EEG_file = [f for f in os.listdir(os.path.join(EEG_path, folder)) if 'BBO-epo' in f]
        Behavioural_file = [f for f in os.listdir(os.path.join(EMG_path, EMG_to_extract)) if 'behavioural_data' in f]
        EEG_filenames.append(os.path.join(EEG_path, EEG_folders[idx], EEG_file[0]))
        EMG_filenames.append(os.path.join(EMG_path, EMG_to_extract, EMG_file[0]))
        Behavioural_filenames.append(os.path.join(EMG_path, EMG_to_extract, Behavioural_file[0]))
    return EEG_filenames, EMG_filenames, Behavioural_filenames



######################### Custom functions to compute coherence #########################
def Compute_coherence_multitaper(x, y, fmin = 0, fmax = 60, fs = 1000, bw = 4):
    # Compute multitaper PSD for both signals (x and y)
    psd_x, freqs, weights_x = psd_array_multitaper(x, sfreq=fs, bandwidth=bw, fmin=fmin, fmax=fmax, adaptive=True, output='complex', normalization='full')
    psd_y, _, weights_y = psd_array_multitaper(y, sfreq=fs, bandwidth=bw, fmin=fmin, fmax=fmax, adaptive=True, output='complex', normalization='full')

    # Compute the cross-spectral density (CSD) between x and y
    csd_xy = np.mean(psd_x * np.conj(psd_y), axis=0)  # CSD

    # # Compute the power spectral densities for x and y
    psd_x_avg = np.mean(np.abs(psd_x)**2, axis=0)
    psd_y_avg = np.mean(np.abs(psd_y)**2, axis=0)

    # # Compute the coherence
    coherence = np.abs(csd_xy)**2 / (psd_x_avg * psd_y_avg)
    return freqs, coherence

def Compute_coherence_nitime(x, y, fs, bw):
    # Stack the two signals together
    combined_data = np.vstack([x, y])

    # Create a TimeSeries object
    time_series = ts.TimeSeries(combined_data, sampling_rate=fs)

    # Set the parameters for multitaper coherence
    bandwidth = bw  # Bandwidth of the multitaper window

    # Initialize the MTCoherenceAnalyzer
    coherence_analyzer = nta.MTCoherenceAnalyzer(time_series, bandwidth=bandwidth)

    # Compute coherence
    coherence = coherence_analyzer.coherence

    # Get frequencies for plotting
    frequencies = coherence_analyzer.frequencies
    coherence = coherence[0, 1]
    return frequencies, coherence







