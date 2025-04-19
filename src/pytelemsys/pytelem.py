import pandas as pd
import numpy as np
import warnings
from typing import Callable, Optional

from pytelemsys.pytrack import TrackData

from pytelemsys.utils import resample_data, low_pass_filter, darboux_to_cartesian


class TelemetryData:
    """Data class for storing telemetry data."""

    def __init__(
        self,
        telem_data_path: str,
        separator: str = "\t",
        comment: str = "#",
        decimal: str = ".",
        fun_conversion: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> None:

        # Read telemetry data from file
        self.data = pd.read_csv(
            telem_data_path, sep=separator, comment=comment, decimal=decimal
        )

        # Apply conversion function if provided
        if fun_conversion is not None:
            self.data = fun_conversion(self.data)

        # Raise a warning fi s & n are not in the data
        if "n" not in self.data or "s" not in self.data:
            warnings.warn("Missing curvilinear coordinates", UserWarning)

    def resample(
        self,
        ref_column: str = "time",
        freq: float = 100,
    ) -> pd.DataFrame:
        """Resample the data."""

        return resample_data(self.data, ref_column=ref_column, freq=freq)

    def compute_curvilinear_coordinates(
        self, track_data: TrackData, xTrj: np.ndarray, yTrj: np.ndarray
    ) -> None:
        """Compute curvilinear coordinates
        :param track_data: TrackData object
        :param xTrj: X trajectory
        :param yTrj: Y trajectory
        """
        # Compute curvilinear coordinates
        pass

    def compute_vehicle_borders(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        theta: np.ndarray,
        banking: np.ndarray,
        slope: np.ndarray,
        VehHalf: float,
    ) -> list:
        """Compute vehicle borders in 3D space.
        :param x: x coordinates of the vehicle.
        :param y: y coordinates of the vehicle.
        :param z: z coordinates of the vehicle.
        :param theta: angle of the vehicle.
        :param banking: banking angle of the vehicle.
        :param slope: slope angle of the vehicle.
        :param VehHalf: half of the vehicle width.
        :return: x, y, z coordinates of the vehicle borders.
        """

        x_R, y_R, z_R = darboux_to_cartesian(x, y, z, theta, banking, slope, -VehHalf)
        x_L, y_L, z_L = darboux_to_cartesian(x, y, z, theta, banking, slope, VehHalf)

        return x_R, y_R, z_R, x_L, y_L, z_L
