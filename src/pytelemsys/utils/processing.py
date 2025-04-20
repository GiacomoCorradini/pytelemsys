import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

#   ____                                 _ _
#  |  _ \ ___  ___  __ _ _ __ ___  _ __ | (_)_ __   __ _
#  | |_) / _ \/ __|/ _` | '_ ` _ \| '_ \| | | '_ \ / _` |
#  |  _ <  __/\__ \ (_| | | | | | | |_) | | | | | | (_| |
#  |_| \_\___||___/\__,_|_| |_| |_| .__/|_|_|_| |_|\__, |
#                                 |_|              |___/


def resample_data(
    df_data_origin: pd.DataFrame, ref_column: str = "time", freq: float = 1.0
) -> pd.DataFrame:
    """
    Resample the data based on the given frequency.

    Args:
        df_data_origin: Original DataFrame to be resampled.
        ref_column: Column to be used as the reference for time. Default is "time".
        freq: Resampling frequency in Hz. Default is 1.0 Hz.

    Returns:
        pd.DataFrame: Resampled DataFrame.
    """

    df_data = df_data_origin.copy()

    # Convert frequency to sampling time in milliseconds
    ts = 1.0 / freq

    if not df_data.index.name == ref_column:
        df_data[ref_column] = pd.to_timedelta(df_data[ref_column], unit="s")
        df_data.set_index(ref_column, inplace=True)
    df_data_resampled = df_data.resample(f"{ts}s").mean().interpolate(method="linear")
    df_data_resampled = df_data_resampled.reset_index()
    df_data_resampled[ref_column] = df_data_resampled[ref_column].dt.total_seconds()

    return df_data_resampled


#   _____ _ _ _            _
#  |  ___(_) | |_ ___ _ __(_)_ __   __ _
#  | |_  | | | __/ _ \ '__| | '_ \ / _` |
#  |  _| | | | ||  __/ |  | | | | | (_| |
#  |_|   |_|_|\__\___|_|  |_|_| |_|\__, |
#                                  |___/


def moving_average(data: list | np.ndarray, window_size: int) -> np.ndarray:
    """calculate the moving average of a signal

    Args:
        data: data to be filtered
        window_size: size of the window

    Returns:
        np.ndarray: filtered data
    """

    return np.convolve(data, np.ones(window_size) / window_size, mode="same")


def low_pass_filter(
    data: np.ndarray, time: np.ndarray, cutoff: float, order: int = 4
) -> np.ndarray:
    """Apply a low-pass Butterworth filter using filtfilt.

    Args:
        data: data to be filtered
        time: time vector
        cutoff: cutoff frequency of the filter in Hz
        order: order of the filter. Default is 4.

    Returns:
        np.ndarray: filtered data
    """

    # Sampling frequency
    fs = 1 / np.mean(np.diff(time))

    # Nyquist frequency
    nyquist = 0.5 * fs

    # Normalized cutoff frequency
    normal_cutoff = cutoff / nyquist

    # Design a Butterworth low-pass filter
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    # Apply the filter using filtfilt
    filtered_data = filtfilt(b, a, data)

    return filtered_data
