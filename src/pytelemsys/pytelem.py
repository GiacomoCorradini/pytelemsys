import pandas as pd
import numpy as np
import warnings
from typing import Callable, Optional

from pytelemsys.utils.track import Track

from pytelemsys.utils import (
    resample_data,
    low_pass_filter,
    darboux_to_cartesian,
    compute_curvilinear_coordinates,
)


class TelemetryData:
    """Data class for storing telemetry data."""

    def __init__(
        self,
        telem_data_path: str = None,
        separator: str = "\t",
        comment: str = "#",
        decimal: str = ".",
        fun_conversion: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> None:
        """Constructor for TelemetryData class.

        :param telem_data_path: Path to the telemetry data file.
        :param separator: Separator used in the telemetry data file, defaults to "\t".
        :param comment: Comment character in the telemetry data file, defaults to "#".
        :param decimal: Decimal character in the telemetry data file, defaults to ".".
        :param fun_conversion: Function to convert the data.
        """

        if telem_data_path is not None:
            self.laod_telem_data(
                telem_data_path,
                separator=separator,
                comment=comment,
                decimal=decimal,
                fun_conversion=fun_conversion,
            )

        else:
            warnings.warn("No telemetry data path provided.", UserWarning)
            self.data = None

    def laod_telem_data(
        self,
        telem_data_path: str = None,
        separator: str = "\t",
        comment: str = "#",
        decimal: str = ".",
        fun_conversion: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> None:
        """Load telemetry data from a file.

        :param telem_data_path: Path to the telemetry data file.
        :param separator: Separator used in the telemetry data file, defaults to "\t".
        :param comment: Comment character in the telemetry data file, defaults to "#".
        :param decimal: Decimal character in the telemetry data file, defaults to ".".
        :param fun_conversion: Function to convert the data.
        """
        # Read telemetry data from file
        self.data = pd.read_csv(
            telem_data_path, sep=separator, comment=comment, decimal=decimal
        )

        # Apply conversion function if provided
        if fun_conversion is not None:
            self.data = fun_conversion(self.data)

        # Raise a warning if s & n are not in the data
        if "n" not in self.data or "s" not in self.data:
            warnings.warn("Missing curvilinear coordinates", UserWarning)

    def assign_telem_data(
        self,
        data: pd.DataFrame,
    ) -> None:
        """Assign telemetry data to the object.

        :param data: DataFrame containing telemetry data.
        """
        self.data = data

    def resample(
        self,
        ref_column: str = "time",
        freq: float = 100,
    ) -> pd.DataFrame:
        """Resample the data.

        :param ref_column: Reference column for resampling.
        :param freq: Frequency for resampling.
        :return: Resampled DataFrame.
        """

        return resample_data(self.data, ref_column=ref_column, freq=freq)

    def compute_curvilinear(
        self, track_data: Track, xTrj: np.ndarray, yTrj: np.ndarray
    ) -> None:
        """Compute curvilinear coordinates.

        :param track_data: Track object.
        :param xTrj: X trajectory.
        :param yTrj: Y trajectory.
        """
        # Validate input lengths
        if len(xTrj) != len(yTrj):
            raise ValueError("xTrj and yTrj must have the same length.")

        # Compute and add curvilinear coordinates to the data
        self.data["s"], self.data["n"] = compute_curvilinear_coordinates(
            track_data, xTrj, yTrj
        )

    def compute_vehicle_borders(
        self,
        x: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        VehHalf: float,
        z: np.ndarray = None,
        banking: np.ndarray = None,
        slope: np.ndarray = None,
    ) -> None:
        """Compute vehicle borders in 3D space.

        :param x: x coordinates of the vehicle.
        :param y: y coordinates of the vehicle.
        :param theta: angle of the vehicle.
        :param VehHalf: half of the vehicle width.
        :param z: z coordinates of the vehicle (default: zeros).
        :param banking: banking angle of the vehicle (default: zeros).
        :param slope: slope angle of the vehicle (default: zeros).
        """

        # Ensure z, banking, and slope have the same size as x using default values
        z = np.zeros_like(x) if z is None else z
        banking = np.zeros_like(x) if banking is None else banking
        slope = np.zeros_like(x) if slope is None else slope

        self.data["x_R"], self.data["y_R"], self.data["z_R"] = darboux_to_cartesian(
            x, y, z, theta, banking, slope, -VehHalf
        )
        self.data["x_L"], self.data["y_L"], self.data["z_L"] = darboux_to_cartesian(
            x, y, z, theta, banking, slope, VehHalf
        )
