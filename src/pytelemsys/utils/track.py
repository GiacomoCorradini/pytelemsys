from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(eq=False)
class Track:
    """Track data read from a track file.

    :ivar abscissa: curvilinear abscissa of the mid line in m.
    :ivar curvature: curvature of the mid line in 1/m (positive for left turns).
    :ivar dir_mid_line: heading angle of the mid line in rad.
    :ivar x_mid_line: x coordinates of the mid line in m.
    :ivar y_mid_line: y coordinates of the mid line in m.
    :ivar elevation: elevation of the mid line in m, defaults to zeros.
    :ivar slope: slope angle in rad (positive uphill), defaults to zeros.
    :ivar banking: banking angle in rad (positive: right side up), defaults to zeros.
    :ivar torsion: torsion in 1/m, defaults to zeros.
    :ivar upsilon: upsilon in 1/m, defaults to zeros.
    :ivar width_no_kerbs_L: left width without kerbs in m.
    :ivar width_no_kerbs_R: right width without kerbs in m.
    :ivar width_kerbs_L: left width with kerbs in m, defaults to width_no_kerbs_L.
    :ivar width_kerbs_R: right width with kerbs in m, defaults to width_no_kerbs_R.
    """

    abscissa: np.ndarray
    curvature: np.ndarray
    dir_mid_line: np.ndarray
    x_mid_line: np.ndarray
    y_mid_line: np.ndarray
    elevation: np.ndarray
    slope: np.ndarray
    banking: np.ndarray
    torsion: np.ndarray
    upsilon: np.ndarray
    width_no_kerbs_L: np.ndarray
    width_no_kerbs_R: np.ndarray
    width_kerbs_L: np.ndarray
    width_kerbs_R: np.ndarray

    def __init__(self, data: pd.DataFrame) -> None:
        """Read the track data from a DataFrame.

        :param data: track data, one row per mid line point.
        """

        # 2D track data (mandatory)
        self.abscissa = np.asarray(data["abscissa"].values)
        self.curvature = np.asarray(data["curvature"].values)
        self.dir_mid_line = np.asarray(data["dir_mid_line"].values)
        self.x_mid_line = np.asarray(data["x_mid_line"].values)
        self.y_mid_line = np.asarray(data["y_mid_line"].values)
        self.width_no_kerbs_L = np.asarray(data["width_no_kerbs_L"].values)
        self.width_no_kerbs_R = np.asarray(data["width_no_kerbs_R"].values)

        # 3D track data (optional)
        self.elevation = (
            np.asarray(data["elevation"].values)
            if "elevation" in data
            else np.zeros_like(self.abscissa)
        )
        self.slope = (
            np.asarray(data["slope"].values)
            if "slope" in data
            else np.zeros_like(self.abscissa)
        )
        self.banking = (
            np.asarray(data["banking"].values)
            if "banking" in data
            else np.zeros_like(self.abscissa)
        )
        self.torsion = (
            np.asarray(data["torsion"].values)
            if "torsion" in data
            else np.zeros_like(self.abscissa)
        )
        self.upsilon = (
            np.asarray(data["upsilon"].values)
            if "upsilon" in data
            else np.zeros_like(self.abscissa)
        )

        # Kerbs width (optional)
        self.width_kerbs_L = (
            np.asarray(data["width_kerbs_L"].values)
            if "width_kerbs_L" in data
            else self.width_no_kerbs_L
        )
        self.width_kerbs_R = (
            np.asarray(data["width_kerbs_R"].values)
            if "width_kerbs_R" in data
            else self.width_no_kerbs_R
        )
