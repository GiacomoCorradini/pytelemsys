from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Track:
    """ Data class for storing track data.
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
        """ Initialize a Track object.

        Args:
            data: DataFrame containing the track data.
        """

        # 2D track data (mandatory)
        self.abscissa         = data['abscissa'].values
        self.curvature        = data['curvature'].values
        self.dir_mid_line     = data['dir_mid_line'].values
        self.x_mid_line       = data['x_mid_line'].values
        self.y_mid_line       = data['y_mid_line'].values
        self.width_no_kerbs_L = data['width_no_kerbs_L'].values
        self.width_no_kerbs_R = data['width_no_kerbs_R'].values

        # 3D track data (optional)
        self.elevation = data['elevation'].values if 'elevation' in data else np.zeros_like(self.abscissa)
        self.slope     = data['slope'].values if 'slope' in data else np.zeros_like(self.abscissa)
        self.banking   = data['banking'].values if 'banking' in data else np.zeros_like(self.abscissa)
        self.torsion   = data['torsion'].values if 'torsion' in data else np.zeros_like(self.abscissa)
        self.upsilon   = data['upsilon'].values if 'upsilon' in data else np.zeros_like(self.abscissa)
        
        # Kerbs width (optional)
        self.width_kerbs_L = data['width_kerbs_L'].values if 'width_kerbs_L' in data else self.width_no_kerbs_L
        self.width_kerbs_R = data['width_kerbs_R'].values if 'width_kerbs_R' in data else self.width_no_kerbs_R
