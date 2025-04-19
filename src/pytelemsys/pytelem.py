import pandas as pd
import numpy as np
import warnings
from typing import Callable, Optional

from pytelemsys.pytrack import TrackData

from pytelemsys.utils.processing import (
    resample_data,
    low_pass_filter,
)


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
