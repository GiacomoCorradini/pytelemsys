"""
This module converts telemetry data from the MLT GP2 format to a standardized format
suitable for further analysis and processing.
"""

import pandas as pd
import numpy as np


def mlt_gp2_converter(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the input telemetry DataFrame by renaming columns and calculating derived metrics.

    :param data: A pandas DataFrame containing telemetry data with predefined column names.
    :return: A transformed pandas DataFrame with standardized column names and computed metrics.
    """

    data["V"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)

    data = data.rename(
        columns={
            "xTrj": "x",
            "yTrj": "y",
            "zeta": "s",
            "y__steer": "steering",
            "p__pos": "throttle",
            "p__neg": "brake",
            "a__x": "ax",
            "a__y": "ay",
            "omega__z": "yaw_rate",
        }
    )

    return data
