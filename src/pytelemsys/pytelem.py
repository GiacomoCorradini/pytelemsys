import pandas as pd
import numpy as np

from dataclasses import dataclass


@dataclass
class Telemetry:
    """Data class for storing telemetry data."""

    # Time/spatial data
    time: np.ndarray
    distance: np.ndarray

    # Inputs
    steering: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray

    # Cartesian coordinates
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    # GPS coordinates
    latitude: np.ndarray
    longitude: np.ndarray
    altitude: np.ndarray

    # Linear accelerations
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray

    # Angular velocities
    roll_rate: np.ndarray
    pitch_rate: np.ndarray
    yaw_rate: np.ndarray

    # Velocity
    V: np.ndarray

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize a Telemetry object.

        Args:
            data: DataFrame containing the telemetry data.
        """


def read_telem_data(telem_data_path, origin):
    """
    Read telemetry data from a file and return it as a pandas DataFrame.
    convert GPS data from hexadecimal to decimal format.
    Note: the separator is a semicolon (;) and all the lones starting with #
          are ignored.
    """

    telem_data = pd.read_csv(telem_data_path, sep=";", comment="#", decimal=",")

    # Convert GPS data from hexadecimal to decimal base, with DD format
    latitude, longitude, elevation = convert_gps_data(telem_data)
    telem_data["latitude"] = latitude
    telem_data["longitude"] = longitude
    telem_data["elevation"] = elevation

    # Convert GPS data to ENU coordinates
    xyz_coord = GPS2XYZ_ENU(
        telem_data["longitude"],
        telem_data["latitude"],
        telem_data["elevation"],
        origin[1],
        origin[0],
        origin[2],
    )

    telem_data["x"] = xyz_coord[:, 0]
    telem_data["y"] = xyz_coord[:, 1]
    telem_data["z"] = xyz_coord[:, 2]

    return telem_data
