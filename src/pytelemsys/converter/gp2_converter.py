"""
This module converts telemetry data from the GP2 ECU to a standardized format
suitable for further analysis and processing.
"""

import numpy as np
import pandas as pd


def gp2_converter(telem_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert telemetry data from the GP2 ECU to a standardized format.

    :param telem_data: A pandas DataFrame containing raw telemetry data
                       from the GP2 ECU.
    :return: A pandas DataFrame with processed and standardized telemetry data.
    """

    # Store gps data, they are stored in hexadecimal format
    latitude_HDM = telem_data["CE_ADR_84_Lat"]
    longitude_HDM = telem_data["CE_ADR_85_Long"]
    elevation_HDM = telem_data["CE_ADR_87_Alt"]

    # Convert to decimal format
    # Note: now they are expressed in degrees minutes decimal format
    #       the results are formatted in 10f
    latitude_DM = hex_to_decimal_10f(latitude_HDM)
    longitude_DM = hex_to_decimal_10f(longitude_HDM)

    # Note: the elavation is converted in meters
    telem_data["z"] = [int(x[:-1], 16) / 1e2 for x in elevation_HDM]

    # Convert to decimal degrees format
    telem_data["lat"] = DM_DD_conversion(latitude_DM)
    telem_data["lon"] = DM_DD_conversion(longitude_DM)

    telem_data = telem_data.rename(
        columns={
            "SteerATWheel": "steering",
            "PEDAL": "throttle",
            "F_BRAK": "brake",
            "ACC_X": "ax",
            "ACC_Y": "ay",
            "LAPTIM": "time",
            "VVEH": "V",
        }
    )

    return telem_data


def hex_to_decimal_10f(hex_num) -> np.ndarray:
    """Convert hexadecimal numbers to decimal format

    :param hex_num: list of hexadecimal numbers
    :return: list of decimal numbers in 10f format
    """

    # Convert the list to a NumPy array
    hex_array = np.array(hex_num)

    # Remove the trailing spaces and convert hex to integers
    int_array = np.vectorize(lambda x: int(x[:-1], 16))(hex_array)

    # Format integers into zero-padded strings
    data_converted = np.char.zfill(int_array.astype(str), 10)

    return data_converted


def DM_DD_conversion(decimal_value) -> np.ndarray:
    """Convert degrees minutes decimal to degrees decimal

    :param decimal_value: value in degrees minutes decimal format
    :return: value in degrees decimal format
    """

    dd_list = []
    for i in range(0, len(decimal_value)):

        # Extract degrees and minutes
        degrees = int(decimal_value[i][:3])  # First 3 digits are degrees
        minutes = int(decimal_value[i][3:]) / 1e5  # Next 2 digits are minutes

        # Convert to degrees decimal
        dd_list.append(degrees + minutes / 60.0)

    return dd_list
