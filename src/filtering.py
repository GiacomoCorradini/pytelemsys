import numpy as np
from scipy.signal import butter, filtfilt

def moving_average(data:list|np.ndarray, window_size:int) -> np.ndarray:
    """ calculate the moving average of a signal

    Args:
        data: data to be filtered
        window_size: size of the window

    Returns:
        np.ndarray: filtered data
    """

    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def low_pass_filter(data:np.ndarray, time:np.ndarray, cutoff:float, order:int = 4) -> np.ndarray:
    """ Apply a low-pass Butterworth filter using filtfilt.

    Args:
        data: data to be filtered
        time: time vector
        cutoff: cutoff frequency of the filter in Hz
        order: order of the filter. Default is 4.

    Returns:
        np.ndarray: filtered data
    """

    # Sampling frequency
    fs = 1/np.mean(np.diff(time))

    # Nyquist frequency
    nyquist = 0.5 * fs

    # Normalized cutoff frequency
    normal_cutoff = cutoff / nyquist
    
    # Design a Butterworth low-pass filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter using filtfilt
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data