
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




# Functions to extract lateralisation indices from EEG data
def Extract_Lat_index(this_subject, channel = "Mean"):
    """
    Function to extract the lateralization index for a single participant
    Parameters:
    - this_subject: MNE TFR object with EEG data and events.
    - channel: str, the channel to extract the lateralization index for. Can be "FC", "C", "CP" or "Mean".
    Returns:
    - Mean_lat_index: float, average lateralization index across left and right arms.
    """
    # Construct the left and right chennels by averaging the TFR of the left and right channels respectively
    if channel == "Mean":
        left_data = np.zeros((this_subject.data.shape[0], len(Left_channels), this_subject.data.shape[2], this_subject.data.shape[3]))
        print(left_data.shape) 
        for idx, channel in enumerate(Left_channels):
            left_data[:, idx, :, :] = this_subject.data[:, this_subject.ch_names.index(channel), :, :]
        left_data = np.mean(left_data, axis=1)
        right_data = np.zeros((this_subject.data.shape[0], len(Right_channels), this_subject.data.shape[2], this_subject.data.shape[3]))
        for idx, channel in enumerate(Right_channels):
            right_data[:, idx, :, :] = this_subject.data[:, this_subject.ch_names.index(channel), :, :]
        right_data = np.mean(right_data, axis=1)
    elif channel == "FC":
        left_data = this_subject.data[:, this_subject.ch_names.index('FC1'), :, :]
        right_data = this_subject.data[:, this_subject.ch_names.index('FC2'), :, :]
    elif channel == "C":
        left_data = this_subject.data[:, this_subject.ch_names.index('C3'), :, :]
        right_data = this_subject.data[:, this_subject.ch_names.index('C4'), :, :]
    elif channel == "CP":
        left_data = this_subject.data[:, this_subject.ch_names.index('CP1'), :, :]
        right_data = this_subject.data[:, this_subject.ch_names.index('CP2'), :, :]
    else:
        raise ValueError("Invalid channel name")
    
    # Compute the contralateral and ipsilateral TFRs for the left and right channels
    events = this_subject.events
    event_id = {'Left':10002,'Invalid':10003, 'Right':10004}
    Left_contra = left_data[events[:, 2] == event_id['Right'],:,:]
    Left_ipsi = left_data[events[:, 2] == event_id['Left'],:,:]

    Right_contra = right_data[events[:, 2] == event_id['Left'],:,:]
    Right_ipsi = right_data[events[:, 2] == event_id['Right'],:,:]

    assert Left_contra.shape[0] == Left_ipsi.shape[0] == Right_contra.shape[0] == Right_ipsi.shape[0] == 45, "The number of trials for the contra and ipsi is incorrect"

    Left_contra_mean = Left_contra.mean(axis=0)
    Left_ipsi_mean = Left_ipsi.mean(axis=0)
    Right_contra_mean = Right_contra.mean(axis=0)
    Right_ipsi_mean = Right_ipsi.mean(axis=0)

    # Compute the lateralization index for the left and right channels - [((contralateral - ipsilateral)/ ipsilateral)  x 100]
    Left_lat_index = ((Left_contra_mean - Left_ipsi_mean)/Left_ipsi_mean)*100
    Right_lat_index = ((Right_contra_mean - Right_ipsi_mean)/Right_ipsi_mean)*100
    
    # average this percentage value for the two channels (left and right)
    Mean_lat_index = (Left_lat_index + Right_lat_index)/2
    return Mean_lat_index


# Function to extract the lateralization index for the EMG data
def Extract_Lat_index_EMG(epochs, path,normalise_baseline = True):
    """
    Computes the EMG lateralization index for a given subject.

    Parameters:
    - sample_subject: mne.tfr object with EEG/EMG data and events.
    - event_id: dict mapping event names to IDs (e.g., {'Left': 1, 'Invalid': 3, 'Right': 2}).
    - expected_trials: int, expected number of trials for validation.

    Returns:
    - Mean_lat_index: float, average lateralization index across left and right arms.
    """

    left_data_time = epochs.get_data()[:, epochs.ch_names.index('EMG_left'), :]
    right_data_time = epochs.get_data()[:, epochs.ch_names.index('EMG_right'), :]

    # Normalise to the first second of each epoch
    if normalise_baseline:
        timebegin = np.argmin(np.abs(epochs.times - (-4)))
        timeend = np.argmin(np.abs(epochs.times - (-3)))
        right_mean = right_data_time[:,timebegin:timeend].mean(axis = 1, keepdims=True)
        right_std = right_data_time[:,timebegin:timeend].std(axis = 1, keepdims=True)
        right_data_time = (right_data_time - right_mean)/right_std

        left_mean = left_data_time[:,timebegin:timeend].mean(axis = 1, keepdims=True)
        left_std = left_data_time[:,timebegin:timeend].std(axis = 1, keepdims=True)
        left_data_time = (left_data_time - left_mean)/left_std

    # Compute RMS
    left_rms = np.sqrt(np.mean(left_data_time**2, axis=1))
    right_rms = np.sqrt(np.mean(right_data_time**2, axis=1))

    # Identify outliers using z-score
    z_thresh = 3  # Common threshold (adjustable)
    left_mean, left_std = np.mean(left_rms), np.std(left_rms)
    right_mean, right_std = np.mean(right_rms), np.std(right_rms)
    left_outliers = np.where((np.abs(left_rms - left_mean) / left_std) > z_thresh)[0]
    right_outliers = np.where((np.abs(right_rms - right_mean) / right_std) > z_thresh)[0]
    left_trials2use = np.ones(len(left_rms), dtype=bool)
    left_trials2use[left_outliers] = False
    right_trials2use = np.ones(len(right_rms), dtype=bool)
    right_trials2use[right_outliers] = False
    print(f'Number of outliers identified in left: {len(left_outliers)}')
    print(f'Number of outliers identified in right: {len(right_outliers)}')

    trials2use = {'Left': left_trials2use, 'Right': right_trials2use}
    # save to path
    source_path = path.split('/')[:-1]
    source_path = '/'.join(source_path) + '/'
    path = os.path.join(source_path, 'trials2use.npy')
    np.save(path, trials2use)

    # Then Z normalise both left and right data in the time domain, but without taking those outliers into consideration
    if not normalise_baseline:
        left_mean = left_data_time[left_trials2use,:].reshape(1,-1).mean(axis = 1)
        left_std = left_data_time[left_trials2use,:].reshape(1,-1).std(axis = 1)
        left_data_time = (left_data_time - left_mean)/left_std

        right_mean = right_data_time[right_trials2use,:].reshape(1,-1).mean(axis = 1)
        right_std = right_data_time[right_trials2use,:].reshape(1,-1).std(axis = 1)
        right_data_time = (right_data_time - right_mean)/right_std

    # Replace only the relevant channels with the normalized data
    left_data_time = np.expand_dims(left_data_time, axis=1)
    right_data_time = np.expand_dims(right_data_time, axis=1)

    new_data = np.concatenate([left_data_time, right_data_time], axis = 1)
    epochs_normalised = mne.EpochsArray(new_data, epochs.info, events=epochs.events, event_id=epochs.event_id, tmin=epochs.tmin)

    frequencies = np.arange(3, 51, 1)  # Define frequencies of interest
    n_cycles = frequencies/2
    sample_subject = epochs_normalised.compute_tfr(method = 'morlet', 
                                        freqs = frequencies,
                                        n_cycles = n_cycles,
                                        output='power',
                                        picks='emg',
                                        average=False, 
                                        return_itc=False)

    events = sample_subject.events
    event_id = {'Left': 1, 'Invalid':3, 'Right':2}

    left_data = sample_subject.data[:, sample_subject.ch_names.index('EMG_left'), :, :]
    right_data = sample_subject.data[:, sample_subject.ch_names.index('EMG_right'), :, :]

    assert left_data.shape[0] == right_data.shape[0], "Number of trials in the left and right hand data is not the same"

    Left_contra = left_data[events[:, 2] == event_id['Right'],:,:]
    left_contra_inclusions = left_trials2use[events[:, 2] == event_id['Right']]
    Left_ipsi = left_data[events[:, 2] == event_id['Left'],:,:]
    left_ipsi_inclusions = left_trials2use[events[:, 2] == event_id['Left']]

    Right_contra = right_data[events[:, 2] == event_id['Left'],:,:]
    right_contra_inclusions = right_trials2use[events[:, 2] == event_id['Left']]
    Right_ipsi = right_data[events[:, 2] == event_id['Right'],:,:]
    right_ipsi_inclusions = right_trials2use[events[:, 2] == event_id['Right']]

    assert Left_contra.shape[0] == Left_ipsi.shape[0] == Right_contra.shape[0] == Right_ipsi.shape[0] == 45, "The number of trials for the contra and ipsi is incorrect"

    Left_contra_mean = Left_contra[left_contra_inclusions,:,:].mean(axis=0)
    Left_ipsi_mean = Left_ipsi[left_ipsi_inclusions,:,:].mean(axis=0)
    Right_contra_mean = Right_contra[right_contra_inclusions,:,:].mean(axis=0)
    Right_ipsi_mean = Right_ipsi[right_ipsi_inclusions,:,:].mean(axis=0)

    # Compute the lateralization index for the left and right channels - ((ipsilateral-contralateral)/contralateral) x 100
    Left_lat_index = ((Left_ipsi_mean - Left_contra_mean)/Left_contra_mean)*100
    Right_lat_index = ((Right_ipsi_mean - Right_contra_mean)/Right_contra_mean)*100

    # average this percentage value for the two channels (left and right)
    Mean_lat_index = (Left_lat_index + Right_lat_index)/2
    return Mean_lat_index, Left_lat_index, Right_lat_index

# Function to compute coherence between EMG and each of the contralateral and ipsilateral EEG channels across all trials
def Compute_coherence_contra_ipsi(Subject_EEG, Subject_EMG, tmin, tmax, Left_channels, Right_channels, trials2use):
    fmin = 0
    fmax = 60
    fs = Subject_EEG.info['sfreq']
    bw = 4
    left_EMG = Subject_EMG.crop(tmin=tmin, tmax=tmax).get_data(picks='EMG_left').squeeze()
    right_EMG = Subject_EMG.crop(tmin=tmin, tmax=tmax).get_data(picks='EMG_right').squeeze()

    left_EEG = []
    right_EEG = []
    for chan in Left_channels:
        left_EEG.append(Subject_EEG.crop(tmin=tmin, tmax=tmax).get_data(picks=chan).squeeze())
    for chan in Right_channels:
        right_EEG.append(Subject_EEG.crop(tmin=tmin, tmax=tmax).get_data(picks=chan).squeeze())
    left_EEG = np.array(left_EEG)
    right_EEG = np.array(right_EEG)
    assert Subject_EEG.info['sfreq'] == Subject_EMG.info['sfreq'], "Sampling frequency is not 1000 Hz"

    # For the left arm
    left_contra_coherence = {}
    left_ipsi_coherence = {}
    for channel in range(left_EEG.shape[0]):
        contra_coherence = []
        ipsi_coherence = []
        for trial in range(left_EMG.shape[0]):
            if not trials2use['Left'][trial]:
                continue
            frequencies, coherence = Compute_coherence_multitaper(left_EMG[trial,:], left_EEG[channel,trial,:], fmin=fmin, fmax=fmax, fs = fs, bw=bw)
            ipsi_coherence.append(coherence)

            frequencies, coherence = Compute_coherence_multitaper(left_EMG[trial,:], right_EEG[channel,trial,:], fmin=fmin, fmax=fmax, fs = fs, bw=bw)
            contra_coherence.append(coherence)
        contra_coherence = np.array(contra_coherence)
        ipsi_coherence = np.array(ipsi_coherence)
        left_contra_coherence[Right_channels[channel]] = contra_coherence.mean(axis=0)
        left_ipsi_coherence[Left_channels[channel]] = ipsi_coherence.mean(axis=0)

    # Right Arm
    right_contra_coherence = {}
    right_ipsi_coherence = {}

    for channel in range(right_EEG.shape[0]):
        contra_coherence = []
        ipsi_coherence = []
        for trial in range(right_EMG.shape[0]):
            if not trials2use['Right'][trial]:
                continue
            frequencies, coherence = Compute_coherence_multitaper(right_EMG[trial,:], right_EEG[channel,trial,:], fmin=fmin, fmax=fmax, fs = fs, bw=bw)
            ipsi_coherence.append(coherence)

            frequencies, coherence = Compute_coherence_multitaper(right_EMG[trial,:], left_EEG[channel,trial,:], fmin=fmin, fmax=fmax, fs = fs, bw=bw)
            contra_coherence.append(coherence)
        contra_coherence = np.array(contra_coherence)
        ipsi_coherence = np.array(ipsi_coherence)
        right_contra_coherence[Left_channels[channel]] = contra_coherence.mean(axis=0)
        right_ipsi_coherence[Right_channels[channel]] = ipsi_coherence.mean(axis=0)

    return left_contra_coherence, left_ipsi_coherence, right_contra_coherence, right_ipsi_coherence, frequencies


def Extract_Anticipatory_Coherence(Subject_EEG, Subject_EMG, tmin, tmax, Left_channels, Right_channels, trials2use):
    fmin = 0
    fmax = 60
    fs = Subject_EEG.info['sfreq']
    bw = 4
    left_EMG = Subject_EMG['Left'].crop(tmin=tmin, tmax=tmax).get_data(picks='EMG_left').squeeze()
    left_trials2use = trials2use['Left'][Subject_EMG.events[:,2] == Subject_EMG.event_id['Left']]
    right_EMG = Subject_EMG['Right'].crop(tmin=tmin, tmax=tmax).get_data(picks='EMG_right').squeeze()
    right_trials2use = trials2use['Right'][Subject_EMG.events[:,2] == Subject_EMG.event_id['Right']]

    left_EEG = []
    right_EEG = []
    for chan in Left_channels:
        left_EEG.append(Subject_EEG['Right'].crop(tmin=tmin, tmax=tmax).get_data(picks=chan).squeeze())
    for chan in Right_channels:
        right_EEG.append(Subject_EEG['Left'].crop(tmin=tmin, tmax=tmax).get_data(picks=chan).squeeze())
    left_EEG = np.array(left_EEG)
    right_EEG = np.array(right_EEG)
    assert Subject_EEG.info['sfreq'] == Subject_EMG.info['sfreq'], "Sampling frequency is not 1000 Hz"

    # For the left arm
    left_contra_coherence = {}
    for channel in range(right_EEG.shape[0]):
        contra_coherence = []
        for trial in range(left_EMG.shape[0]):
            if not left_trials2use[trial]:
                continue
            frequencies, coherence = Compute_coherence_multitaper(left_EMG[trial,:], right_EEG[channel,trial,:], fmin=fmin, fmax=fmax, fs = fs, bw=bw)
            contra_coherence.append(coherence)
        contra_coherence = np.array(contra_coherence)
        left_contra_coherence[Right_channels[channel]] = contra_coherence.mean(axis=0)

    # Right Arm
    right_contra_coherence = {} 
    for channel in range(left_EEG.shape[0]):
        contra_coherence = []
        for trial in range(right_EMG.shape[0]):
            if not right_trials2use[trial]:
                continue
            frequencies, coherence = Compute_coherence_multitaper(right_EMG[trial,:], left_EEG[channel,trial,:], fmin=fmin, fmax=fmax, fs = fs, bw=bw)
            contra_coherence.append(coherence)
        contra_coherence = np.array(contra_coherence)
        right_contra_coherence[Left_channels[channel]] = contra_coherence.mean(axis=0)

    return left_contra_coherence, right_contra_coherence, frequencies


# 2) coherence between the EMG channels and the ipsilateral EEG channels for the trials 
# where the stimulation/attention was given to the arm ipsilateral to the EMG. 
def Extract_Coherence_Condition2(Subject_EEG, Subject_EMG, tmin, tmax, Left_channels, Right_channels, trials2use):
    fmin = 0
    fmax = 60
    fs = Subject_EEG.info['sfreq']
    bw = 4
    left_EMG_left_stim = Subject_EMG['Left'].crop(tmin=tmin, tmax=tmax).get_data(picks='EMG_left').squeeze()
    right_EMG_right_stim = Subject_EMG['Right'].crop(tmin=tmin, tmax=tmax).get_data(picks='EMG_right').squeeze()
    left_EMG_right_stim = Subject_EMG['Right'].crop(tmin=tmin, tmax=tmax).get_data(picks='EMG_left').squeeze()
    right_EMG_left_stim = Subject_EMG['Left'].crop(tmin=tmin, tmax=tmax).get_data(picks='EMG_right').squeeze()

    # Extract the trials to be used
    Left_leftStim_trials = trials2use['Left'][Subject_EMG.events[:,2] == Subject_EMG.event_id['Left']]
    Left_rightStim_trials = trials2use['Left'][Subject_EMG.events[:,2] == Subject_EMG.event_id['Right']]
    Right_leftStim_trials = trials2use['Right'][Subject_EMG.events[:,2] == Subject_EMG.event_id['Left']]
    Right_rightStim_trials = trials2use['Right'][Subject_EMG.events[:,2] == Subject_EMG.event_id['Right']]

    left_EEG_left_stim = []
    right_EEG_right_stim = []
    left_EEG_right_stim = []
    right_EEG_left_stim = []
    for chan in Left_channels:
        left_EEG_left_stim.append(Subject_EEG['Left'].crop(tmin=tmin, tmax=tmax).get_data(picks=chan).squeeze())
        left_EEG_right_stim.append(Subject_EEG['Right'].crop(tmin=tmin, tmax=tmax).get_data(picks=chan).squeeze())
    for chan in Right_channels:
        right_EEG_right_stim.append(Subject_EEG['Right'].crop(tmin=tmin, tmax=tmax).get_data(picks=chan).squeeze())
        right_EEG_left_stim.append(Subject_EEG['Left'].crop(tmin=tmin, tmax=tmax).get_data(picks=chan).squeeze())
    left_EEG_left_stim = np.array(left_EEG_left_stim)
    left_EEG_right_stim = np.array(left_EEG_right_stim)
    right_EEG_right_stim = np.array(right_EEG_right_stim)
    right_EEG_left_stim = np.array(right_EEG_left_stim)
    assert Subject_EEG.info['sfreq'] == Subject_EMG.info['sfreq'], "Sampling frequency is not 1000 Hz"

    # Right Arm
    right_ipsi_right_stim = {} 
    right_ipsi_left_stim = {}
    for channel in range(len(Right_channels)):
        ipsi_coherence_right_stim = []
        ipsi_coherence_left_stim = []
        for trial in range(right_EMG_right_stim.shape[0]):
            if Right_rightStim_trials[trial]:
                frequencies, coherence = Compute_coherence_multitaper(right_EMG_right_stim[trial,:], right_EEG_right_stim[channel,trial,:], fmin=fmin, fmax=fmax, fs = fs, bw=bw)
                ipsi_coherence_right_stim.append(coherence)
            if Right_leftStim_trials[trial]:
                frequencies, coherence = Compute_coherence_multitaper(right_EMG_left_stim[trial,:], right_EEG_left_stim[channel,trial,:], fmin=fmin, fmax=fmax, fs = fs, bw=bw)
                ipsi_coherence_left_stim.append(coherence)
        ipsi_coherence_right_stim = np.array(ipsi_coherence_right_stim)
        ipsi_coherence_left_stim = np.array(ipsi_coherence_left_stim)
        right_ipsi_right_stim[Right_channels[channel]] = ipsi_coherence_right_stim.mean(axis=0)
        right_ipsi_left_stim[Right_channels[channel]] = ipsi_coherence_left_stim.mean(axis=0)

    # Left Arm
    left_ipsi_left_stim = {}
    left_ipsi_right_stim = {}
    for channel in range(len(Left_channels)):
        ipsi_coherence_left_stim = []
        ipsi_coherence_right_stim = []
        for trial in range(left_EMG_left_stim.shape[0]):
            if Left_leftStim_trials[trial]:
                frequencies, coherence = Compute_coherence_multitaper(left_EMG_left_stim[trial,:], left_EEG_left_stim[channel,trial,:], fmin=fmin, fmax=fmax, fs = fs, bw=bw)
                ipsi_coherence_left_stim.append(coherence)
            if Left_rightStim_trials[trial]:
                frequencies, coherence = Compute_coherence_multitaper(left_EMG_right_stim[trial,:], left_EEG_right_stim[channel,trial,:], fmin=fmin, fmax=fmax, fs = fs, bw=bw)
                ipsi_coherence_right_stim.append(coherence)
        ipsi_coherence_left_stim = np.array(ipsi_coherence_left_stim)
        ipsi_coherence_right_stim = np.array(ipsi_coherence_right_stim)
        left_ipsi_left_stim[Left_channels[channel]] = ipsi_coherence_left_stim.mean(axis=0)
        left_ipsi_right_stim[Left_channels[channel]] = ipsi_coherence_right_stim.mean(axis=0)

    return right_ipsi_right_stim, right_ipsi_left_stim, left_ipsi_left_stim, left_ipsi_right_stim, frequencies


# Extract trial level binned lateralization 
def Extract_sorted_lateralization(Subject_EEG, Subject_EMG, tmin, tmax, fmin, fmax, Left_channels, Right_channels, trials2use, normalise_baseline = True, outlier_threshold = 3):
    # Extract average power for the left and right EEG channels and average together

    # Normalise to the first second of each epoch
    if normalise_baseline:
        left_data_time = Subject_EMG.get_data()[:, Subject_EMG.ch_names.index('EMG_left'), :]
        right_data_time = Subject_EMG.get_data()[:, Subject_EMG.ch_names.index('EMG_right'), :]

        timebegin = np.argmin(np.abs(Subject_EMG.times - (-4)))
        timeend = np.argmin(np.abs(Subject_EMG.times - (-3)))
        right_mean = right_data_time[:,timebegin:timeend].mean(axis = 1, keepdims=True)
        right_std = right_data_time[:,timebegin:timeend].std(axis = 1, keepdims=True)
        right_data_time = (right_data_time - right_mean)/right_std

        left_mean = left_data_time[:,timebegin:timeend].mean(axis = 1, keepdims=True)
        left_std = left_data_time[:,timebegin:timeend].std(axis = 1, keepdims=True)
        left_data_time = (left_data_time - left_mean)/left_std

        # Compute RMS
        left_rms = np.sqrt(np.mean(left_data_time**2, axis=1))
        right_rms = np.sqrt(np.mean(right_data_time**2, axis=1))

        # Identify outliers using z-score
        z_thresh = outlier_threshold  # Common threshold (adjustable)
        left_mean, left_std = np.mean(left_rms), np.std(left_rms)
        right_mean, right_std = np.mean(right_rms), np.std(right_rms)
        left_outliers = np.where((np.abs(left_rms - left_mean) / left_std) > z_thresh)[0]
        right_outliers = np.where((np.abs(right_rms - right_mean) / right_std) > z_thresh)[0]
        left_trials2use = np.ones(len(left_rms), dtype=bool)
        left_trials2use[left_outliers] = False
        right_trials2use = np.ones(len(right_rms), dtype=bool)
        right_trials2use[right_outliers] = False

        # Put the data back into the epochs
        Subject_EMG._data[:, Subject_EMG.ch_names.index('EMG_left'), :] = left_data_time
        Subject_EMG._data[:, Subject_EMG.ch_names.index('EMG_right'), :] = right_data_time

    Left_EEG_power = Subject_EEG.compute_psd(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=Left_channels).get_data().mean(axis=1).mean(axis=1).squeeze()
    Right_EEG_power = Subject_EEG.compute_psd(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=Right_channels).get_data().mean(axis=1).mean(axis=1).squeeze()

    # Extract average power for the left and right EMG channels and average together
    Left_EMG_power = Subject_EMG.compute_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks='EMG_left').get_data(picks='EMG_left').mean(axis=2).squeeze()
    Right_EMG_power = Subject_EMG.compute_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks='EMG_right').get_data(picks='EMG_right').mean(axis=2).squeeze()

    # Calculate the normalized difference in beta power between the left and right recording sites
    Cortical_lateralization = (Right_EEG_power - Left_EEG_power) / (Right_EEG_power + Left_EEG_power)
    Muscular_lateralization = (Left_EMG_power - Right_EMG_power) / (Left_EMG_power + Right_EMG_power)

    # Sort the trials according to cortical lateralization (separately for left- and right-cued trials) and place them into five consecutive bins.
    EEG_events = Subject_EEG.events
    EMG_events = Subject_EMG.events

    EEG_event_id = Subject_EEG.event_id
    EMG_event_id = Subject_EMG.event_id

    if not normalise_baseline:
        left_trials2use = trials2use['Left']
        right_trials2use = trials2use['Right']
    combined_trials2use = np.array([l and r for l, r in zip(left_trials2use, right_trials2use)])

    Cortical_lateralization_left = Cortical_lateralization[EEG_events[:,2] == EEG_event_id['Left']]
    Cortical_lateralization_right = Cortical_lateralization[EEG_events[:,2] == EEG_event_id['Right']]

    Muscular_lateralization_left = Muscular_lateralization[EMG_events[:,2] == EMG_event_id['Left']]
    Muscular_lateralization_right = Muscular_lateralization[EMG_events[:,2] == EMG_event_id['Right']]

    trials2use_left = combined_trials2use[EEG_events[:,2] == EEG_event_id['Left']]
    trials2use_right = combined_trials2use[EEG_events[:,2] == EEG_event_id['Right']]

    # Sort cortical lateralisation and get the sorting indices
    Cortical_lateralization_left_sortind = np.argsort(Cortical_lateralization_left)
    Cortical_lateralization_right_sortind = np.argsort(Cortical_lateralization_right)

    # Sort the cortical and muscular lateralisation
    Cortical_lateralization_left_sorted = Cortical_lateralization_left[Cortical_lateralization_left_sortind]
    Cortical_lateralization_right_sorted = Cortical_lateralization_right[Cortical_lateralization_right_sortind]

    Muscular_lateralization_left_sorted = Muscular_lateralization_left[Cortical_lateralization_left_sortind]
    Muscular_lateralization_right_sorted = Muscular_lateralization_right[Cortical_lateralization_right_sortind]

    trials2use_left_sorted = trials2use_left[Cortical_lateralization_left_sortind]
    trials2use_right_sorted = trials2use_right[Cortical_lateralization_right_sortind]

    # Bin the data into 5 consecutive bins
    bins = 5
    Cortical_lateralization_left_binned = np.array_split(Cortical_lateralization_left_sorted, bins)
    Cortical_lateralization_right_binned = np.array_split(Cortical_lateralization_right_sorted, bins)
    Muscular_lateralization_left_binned = np.array_split(Muscular_lateralization_left_sorted, bins)
    Muscular_lateralization_right_binned = np.array_split(Muscular_lateralization_right_sorted, bins)

    trials2use_left_sorted_binned = np.array_split(trials2use_left_sorted, bins)
    trials2use_right_sorted_binned = np.array_split(trials2use_right_sorted, bins)

    # Average the bins
    Cortical_lateralization_left_mean = np.array([np.mean(bin[trials2use_left_sorted_binned[i]]) for i, bin in enumerate(Cortical_lateralization_left_binned)])
    Cortical_lateralization_right_mean = np.array([np.mean(bin[trials2use_right_sorted_binned[i]]) for i, bin in enumerate(Cortical_lateralization_right_binned)])
    Muscular_lateralization_left_mean = np.array([np.mean(bin[trials2use_left_sorted_binned[i]]) for i, bin in enumerate(Muscular_lateralization_left_binned)])
    Muscular_lateralization_right_mean = np.array([np.mean(bin[trials2use_right_sorted_binned[i]]) for i, bin in enumerate(Muscular_lateralization_right_binned)])

    return Cortical_lateralization_left_mean, Cortical_lateralization_right_mean, Muscular_lateralization_left_mean, Muscular_lateralization_right_mean
