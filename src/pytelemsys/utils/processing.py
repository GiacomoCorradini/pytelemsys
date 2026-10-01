import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

#   ____                                 _ _
#  |  _ \ ___  ___  __ _ _ __ ___  _ __ | (_)_ __   __ _
#  | |_) / _ \/ __|/ _` | '_ ` _ \| '_ \| | | '_ \ / _` |
#  |  _ <  __/\__ \ (_| | | | | | | |_) | | | | | | (_| |
#  |_| \_\___||___/\__,_|_| |_| |_| .__/|_|_|_| |_|\__, |
#                                 |_|              |___/


def resample_data(
    df_data_origin: pd.DataFrame, ref_column: str = "time", freq: float = 1.0
) -> pd.DataFrame:
    """Resample the data on a uniform time grid using linear interpolation.

    Non-numeric columns are dropped. When downsampling, low-pass filter the
    data first to avoid aliasing.

    :param df_data_origin: DataFrame to be resampled.
    :param ref_column: column with the time in s, defaults to "time".
    :param freq: resampling frequency in Hz, defaults to 1.0.
    :return: resampled DataFrame.
    """

    df_data = df_data_origin.select_dtypes("number")
    time = df_data[ref_column].to_numpy()

    if np.any(np.diff(time) <= 0):
        raise ValueError(f"'{ref_column}' must be strictly increasing")

    # Uniform time grid starting at the first sample
    ts = 1.0 / freq
    n_samples = int(np.floor((time[-1] - time[0]) / ts)) + 1
    time_resampled = time[0] + np.arange(n_samples) * ts

    return pd.DataFrame(
        {col: np.interp(time_resampled, time, df_data[col]) for col in df_data}
    )


#   _____ _ _ _            _
#  |  ___(_) | |_ ___ _ __(_)_ __   __ _
#  | |_  | | | __/ _ \ '__| | '_ \ / _` |
#  |  _| | | | ||  __/ |  | | | | | (_| |
#  |_|   |_|_|\__\___|_|  |_|_| |_|\__, |
#                                  |___/


def moving_average(data: list | np.ndarray, window_size: int) -> np.ndarray:
    """Compute the moving average of a signal.

    :param data: data to be filtered.
    :param window_size: size of the window (odd, to avoid a half-sample shift).
    :return: filtered data.
    """

    if window_size > len(data):
        raise ValueError("window_size must not exceed the length of data")

    # Divide by the number of samples in each window, so the edges are not attenuated
    kernel = np.ones(window_size)
    return np.convolve(data, kernel, mode="same") / np.convolve(
        np.ones(len(data)), kernel, mode="same"
    )


def low_pass_filter(
    data: np.ndarray, time: np.ndarray, cutoff: float, order: int = 4
) -> np.ndarray:
    """Apply a zero-phase low-pass Butterworth filter (filtfilt).

    :param data: data to be filtered.
    :param time: time vector in s.
    :param cutoff: cutoff frequency in Hz.
    :param order: order of the filter, defaults to 4.
    :return: filtered data.
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


def savitzky_golay_filter(
    data: np.ndarray, window_length: int = 21, polyorder: int = 3
) -> np.ndarray:
    """Apply a Savitzky-Golay filter to the data.

    :param data: data to be filtered.
    :param window_length: window length (odd, greater than polyorder), defaults to 21.
    :param polyorder: order of the fitting polynomial, defaults to 3.
    :return: filtered data.
    """

    filtered_data = savgol_filter(
        data, window_length=window_length, polyorder=polyorder
    )

    return filtered_data
