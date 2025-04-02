import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

#   ____                                 _ _             
#  |  _ \ ___  ___  __ _ _ __ ___  _ __ | (_)_ __   __ _ 
#  | |_) / _ \/ __|/ _` | '_ ` _ \| '_ \| | | '_ \ / _` |
#  |  _ <  __/\__ \ (_| | | | | | | |_) | | | | | | (_| |
#  |_| \_\___||___/\__,_|_| |_| |_| .__/|_|_|_| |_|\__, |
#                                 |_|              |___/ 

def resample_data(df_data_origin: pd.DataFrame, ts:str = '1') -> pd.DataFrame:
    """
    Resample the data.
    """

    df_data = df_data_origin.copy()

    df_data['time'] = pd.to_timedelta(df_data['time'], unit='s')
    df_data.set_index('time', inplace=True)
    df_data_resampled = df_data.resample(ts + 'ms').mean().interpolate(method='linear')
    df_data_resampled = df_data_resampled.reset_index()
    df_data_resampled['time'] = df_data_resampled['time'].dt.total_seconds()

    return df_data_resampled

#   _____ _ _ _            _             
#  |  ___(_) | |_ ___ _ __(_)_ __   __ _ 
#  | |_  | | | __/ _ \ '__| | '_ \ / _` |
#  |  _| | | | ||  __/ |  | | | | | (_| |
#  |_|   |_|_|\__\___|_|  |_|_| |_|\__, |
#                                  |___/ 

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

#   _____     _   _                 _   _             
#  | ____|___| |_(_)_ __ ___   __ _| |_(_) ___  _ __  
#  |  _| / __| __| | '_ ` _ \ / _` | __| |/ _ \| '_ \ 
#  | |___\__ \ |_| | | | | | | (_| | |_| | (_) | | | |
#  |_____|___/\__|_|_| |_| |_|\__,_|\__|_|\___/|_| |_|
                                                    
def estimate_theta(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    """ Estimate the angle of a 2D curve.

    Args:
        x: x coordinates of the curve.
        y: y coordinates of the curve.

    Returns:
        np.ndarray: angle of the curve.
    """

    # First derivatives (central difference)
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Angle formula
    theta = np.arctan2(dy, dx)

    # Append the last value to have the same size
    theta = np.append(theta, theta[-1])

    return theta

def estimate_curvature(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    """ Estimate the curvature of a 2D curve.

    Args:
        x: x coordinates of the curve.
        y: y coordinates of the curve.

    Returns:
        np.ndarray: curvature of the curve.
    """

    # First derivatives (central difference)
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Second derivatives (central difference)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

    return curvature

#    ____                              _             
#   / ___|___  _ ____   _____ _ __ ___(_) ___  _ __  
#  | |   / _ \| '_ \ \ / / _ \ '__/ __| |/ _ \| '_ \ 
#  | |__| (_) | | | \ V /  __/ |  \__ \ | (_) | | | |
#   \____\___/|_| |_|\_/ \___|_|  |___/_|\___/|_| |_|
                                                   
def darboux_to_cartesian(x_ref:float, y_ref:float, z_ref:float, theta_ref:float, bank_ref:float, slope_ref:float, n:float) -> tuple[float, float, float]:
    """ Convert Darboux frame to Cartesian

    Args:
        x_ref: x coordinate of the reference point.
        y_ref: y coordinate of the reference point.
        z_ref: z coordinate of the reference point.
        theta_ref: angle of the reference point.
        bank_ref: banking of the reference point.
        slope_ref: slope of the reference point.
        n: lateral distance from the reference point.

    Returns:
        x, y, z: Cartesian coordinates.
    """

    s_bank  = np.sin(bank_ref)
    c_bank  = np.cos(bank_ref)
    s_slope = np.sin(slope_ref)
    c_slope = np.cos(slope_ref)
    s_theta = np.sin(theta_ref)
    c_theta = np.cos(theta_ref)

    x = x_ref - n * s_theta * c_bank + c_theta * s_slope * s_bank
    y = y_ref + n * c_theta * c_bank + s_theta * s_slope * s_bank
    z = z_ref + n * c_slope * s_bank

    return x, y, z

def GPS2XYZ_ENU(longitude:np.ndarray, latitude:np.ndarray, altitude:np.ndarray, origin:tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Convert GPS coordinates to ENU coordinates

    Args:
        longitude: longitude of the point.
        latitude: latitude of the point.
        altitude: altitude of the point.
        origin: origin of the ENU coordinates.

    Returns:
        tuple: (x,y,z) ENU coordinates.
    """

    # Origin coordinates
    lon_0 = origin[0]
    lat_0 = origin[1]
    alt0 = origin[2]

    # WGS 84 parameters
    a = 6378137.0  # Semi-major axis
    b = 6356752.31424518  # Semi-minor axis
    f_inv = 298.257223563  # Inverse flattening
    e = 0.0818191908426215  # Eccentricity

    # Calculate squared eccentricity
    e2 = e**2

    # Convert to radians
    lat = np.radians(latitude)
    lon = np.radians(longitude)
    lat0 = np.radians(lat_0)
    lon0 = np.radians(lon_0)

    # Radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat0)**2)

    # Calculate normalized Cartesian coordinates
    norm_x = np.cos(lat) * np.cos(lon)
    norm_y = np.cos(lat) * np.sin(lon)
    norm_z = np.sin(lat)

    # Cartesian coordinates of point P in ECEF
    PX = norm_x * (N + altitude)
    PY = norm_y * (N + altitude)
    PZ = norm_z * ((1 - e2) * N + altitude)

    # Cartesian coordinates of the origin in ECEF
    PX0 = np.cos(lon0) * np.cos(lat0) * (N + alt0)
    PY0 = np.sin(lon0) * np.cos(lat0) * (N + alt0)
    PZ0 = np.sin(lat0) * ((1 - e2) * N + alt0)

    # East, North, Up unit vectors at origin in ECEF
    uvec_E0 = np.array([-np.sin(lon0), np.cos(lon0), 0])
    uvec_N0 = np.array([
        -np.cos(lon0) * np.sin(lat0),
        -np.sin(lon0) * np.sin(lat0),
        np.cos(lat0)
    ])
    uvec_U0 = np.array([
        np.cos(lon0) * np.cos(lat0),
        np.sin(lon0) * np.cos(lat0),
        np.sin(lat0)
    ])

    # Position vectors
    P = np.stack((PX, PY, PZ), axis=-1)
    origin = np.array([PX0, PY0, PZ0])

    # Projection on tangent plane
    DP = P - origin
    XYZ = np.stack((
        np.dot(DP, uvec_E0),
        np.dot(DP, uvec_N0),
        np.dot(DP, uvec_U0)
    ), axis=-1)

    return XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

def convert_gps_data(telem_data):
    """
    Convert GPS telemetry data stored in hexadecimal base and DM format.

    Args: telem_data (pandas DataFrame): telemetry data.

    Returns: latitude_DD, longitude_DD, elevation_M (np.array): GPS data in decimal base and DD format.
    """

    # Store gps data, they are stored in hexadecimal format
    latitude_HDM = telem_data['CE_ADR_84_Lat']
    longitude_HDM = telem_data['CE_ADR_85_Long']
    elevation_HDM = telem_data['CE_ADR_87_Alt']

    # Convert to decimal format
    # Note: now they are expressed in degrees minutes decimal format
    #       the results are formatted in 10f
    latitude_DM = hex_to_decimal_10f(latitude_HDM)
    longitude_DM = hex_to_decimal_10f(longitude_HDM)

    # Note: the elavation is converted in meters
    elevation_M = [int(x[:-1], 16)/1e2 for x in elevation_HDM]

    # Convert to decimal degrees format
    latitude_DD = DM_DD_conversion(latitude_DM)
    longitude_DD = DM_DD_conversion(longitude_DM)

    return latitude_DD, longitude_DD, elevation_M

def hex_to_decimal_10f(hex_num):
    """
    Convert a hexadecimal number with 'h' suffix to decimal base.

    Args: hexa_num (list): list of hexadecimal numbers.

    Returns: data_converted (list): list of decimal numbers.
    """
    # Convert the list to a NumPy array
    hex_array = np.array(hex_num)

    # Remove the trailing spaces and convert hex to integers
    int_array = np.vectorize(lambda x: int(x[:-1], 16))(hex_array)

    # Format integers into zero-padded strings
    data_converted = np.char.zfill(int_array.astype(str), 10)

    return data_converted

def DM_DD_conversion(decimal_value):
    """
        Convert a degrees minutes decimal format to decimal degrees.
    """

    dd_list = []
    for i in range(0, len(decimal_value)):
        
        # Extract degrees and minutes
        degrees = int(decimal_value[i][:3]) # First 3 digits are degrees
        minutes = int(decimal_value[i][3:])/1e5 # Next 2 digits are minutes

        # Convert to degrees decimal
        dd_list.append(degrees + minutes/60.0)

    return dd_list
