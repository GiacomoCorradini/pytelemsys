import pandas as pd
import numpy as np
import warnings
from typing import Callable, Optional

from pytelemsys.utils.track import Track

from pytelemsys.utils import (
    resample_data,
    darboux_to_cartesian,
    cartesian_to_curvilinear,
)


class TelemetryData:
    """Telemetry data stored in a DataFrame."""

    def __init__(
        self,
        telem_data_path: str | None = None,
        separator: str = "\t",
        comment: str = "#",
        decimal: str = ".",
        fun_conversion: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> None:
        """Load the telemetry data from a file, if given.

        :param telem_data_path: path of the telemetry file.
        :param separator: column separator, defaults to "\\t".
        :param comment: comment character, defaults to "#".
        :param decimal: decimal separator, defaults to ".".
        :param fun_conversion: function applied to the loaded DataFrame.
        """

        if telem_data_path is not None:
            self.load_telem_data(
                telem_data_path,
                separator=separator,
                comment=comment,
                decimal=decimal,
                fun_conversion=fun_conversion,
            )

        else:
            warnings.warn("No telemetry data path provided.", UserWarning)
            self.data = None

    def load_telem_data(
        self,
        telem_data_path: str | None = None,
        separator: str = "\t",
        comment: str = "#",
        decimal: str = ".",
        fun_conversion: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> None:
        """Load the telemetry data from a file.

        :param telem_data_path: path of the telemetry file.
        :param separator: column separator, defaults to "\\t".
        :param comment: comment character, defaults to "#".
        :param decimal: decimal separator, defaults to ".".
        :param fun_conversion: function applied to the loaded DataFrame.
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
        """Set the telemetry data.

        :param data: telemetry DataFrame.
        """
        self.data = data

    def resample(
        self,
        ref_column: str = "time",
        freq: float = 100,
    ) -> pd.DataFrame:
        """Resample the data on a uniform time grid.

        :param ref_column: column with the time in s, defaults to "time".
        :param freq: resampling frequency in Hz, defaults to 100.
        :return: resampled DataFrame.
        """

        return resample_data(self.data, ref_column=ref_column, freq=freq)

    def compute_curvilinear(
        self, track_data: Track, x: np.ndarray, y: np.ndarray
    ) -> None:
        """Add the curvilinear coordinates s and n to the data.

        :param track_data: track data.
        :param x: x coordinates of the trajectory.
        :param y: y coordinates of the trajectory.
        """
        # Validate input lengths
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")

        # Compute and add curvilinear coordinates to the data
        self.data["s"], self.data["n"] = cartesian_to_curvilinear(track_data, x, y)

    def compute_vehicle_borders(
        self,
        x: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        half_width: float,
        z: np.ndarray = None,
        banking: np.ndarray = None,
        slope: np.ndarray = None,
    ) -> None:
        """Add the left and right vehicle borders to the data.

        :param x: x coordinates of the vehicle.
        :param y: y coordinates of the vehicle.
        :param theta: heading angle of the vehicle.
        :param half_width: half width of the vehicle.
        :param z: z coordinates of the vehicle, defaults to zeros.
        :param banking: banking angle, defaults to zeros.
        :param slope: slope angle, defaults to zeros.
        """

        # Ensure z, banking, and slope have the same size as x using default values
        z = np.zeros_like(x) if z is None else z
        banking = np.zeros_like(x) if banking is None else banking
        slope = np.zeros_like(x) if slope is None else slope

        self.data["x_R"], self.data["y_R"], self.data["z_R"] = darboux_to_cartesian(
            x, y, z, theta, banking, slope, -half_width
        )
        self.data["x_L"], self.data["y_L"], self.data["z_L"] = darboux_to_cartesian(
            x, y, z, theta, banking, slope, half_width
        )

    def save_data(
        self,
        file_path: str,
        separator: str = "\t",
        index: bool = False,
    ) -> None:
        """Save the telemetry data to a file.

        :param file_path: path of the output file.
        :param separator: column separator, defaults to "\\t".
        :param index: write the DataFrame index, defaults to False.
        """
        self.data.to_csv(file_path, sep=separator, index=index)
