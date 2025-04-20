from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Track:
    """Data class for storing track data.

    Attributes:
        abscissa: Array of abscissa values representing the track's longitudinal position.
        curvature: Array of curvature values along the track.
        dir_mid_line: Array of direction values for the midline of the track.
        x_mid_line: Array of x-coordinates for the midline of the track.
        y_mid_line: Array of y-coordinates for the midline of the track.
        elevation: Array of elevation values along the track. Defaults to zeros if not provided.
        slope: Array of slope values along the track. Defaults to zeros if not provided.
        banking: Array of banking values along the track. Defaults to zeros if not provided.
        torsion: Array of torsion values along the track. Defaults to zeros if not provided.
        upsilon: Array of upsilon values along the track. Defaults to zeros if not provided.
        width_no_kerbs_L: Array of left-side track widths without kerbs.
        width_no_kerbs_R: Array of right-side track widths without kerbs.
        width_kerbs_L: Array of left-side track widths with kerbs. Defaults to `width_no_kerbs_L` if not provided.
        width_kerbs_R: Array of right-side track widths with kerbs. Defaults to `width_no_kerbs_R` if not provided.
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
        """Initialize a Track object.

        Args:
            data: DataFrame containing the track data.
        """

        # 2D track data (mandatory)
        self.abscissa = data["abscissa"].values
        self.curvature = data["curvature"].values
        self.dir_mid_line = data["dir_mid_line"].values
        self.x_mid_line = data["x_mid_line"].values
        self.y_mid_line = data["y_mid_line"].values
        self.width_no_kerbs_L = data["width_no_kerbs_L"].values
        self.width_no_kerbs_R = data["width_no_kerbs_R"].values

        # 3D track data (optional)
        self.elevation = (
            data["elevation"].values
            if "elevation" in data
            else np.zeros_like(self.abscissa)
        )
        self.slope = (
            data["slope"].values if "slope" in data else np.zeros_like(self.abscissa)
        )
        self.banking = (
            data["banking"].values
            if "banking" in data
            else np.zeros_like(self.abscissa)
        )
        self.torsion = (
            data["torsion"].values
            if "torsion" in data
            else np.zeros_like(self.abscissa)
        )
        self.upsilon = (
            data["upsilon"].values
            if "upsilon" in data
            else np.zeros_like(self.abscissa)
        )

        # Kerbs width (optional)
        self.width_kerbs_L = (
            data["width_kerbs_L"].values
            if "width_kerbs_L" in data
            else self.width_no_kerbs_L
        )
        self.width_kerbs_R = (
            data["width_kerbs_R"].values
            if "width_kerbs_R" in data
            else self.width_no_kerbs_R
        )
