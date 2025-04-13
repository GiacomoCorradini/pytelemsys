import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import re
from dataclasses import dataclass

from pytelemsys.utils.conversion import darboux_to_cartesian


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


class TrackData:
    """
    TrackData class stores the track data and the origin of the track.
    """

    def __init__(self, track_data_path: str) -> None:
        """Initialize the TrackData class.

        Args:
            track_data_path: the path of the track data file.
        """

        # Save the path of the track data
        self.track_data_path = track_data_path

        # Read the origin of the track
        self.origin = self._read_track_origin(track_data_path)

        # Read the track data
        self.track = self._read_track_data(track_data_path)

    def plot_track_2D(self, ax: plt.figure, plot_kerbs: bool = False) -> None:
        """Plot the 2D representation of the track.

        Args:
            ax: the axis of the plot.
            plot_kerbs: flag to plot the kerbs.
        """

        # Extract left and right margin points
        x, y = self.track.x_mid_line, self.track.y_mid_line

        # Plot the mid line
        ax.plot(x, y, "k-", linewidth=0.5, label="Mid line")

        # Extract left and right margin points
        xL, yL = self.track.x_margin_no_kerb_L, self.track.y_margin_no_kerb_L
        xR, yR = self.track.x_margin_no_kerb_R, self.track.y_margin_no_kerb_R

        # Plot track margins
        ax.plot(xL, yL, "k-", linewidth=1, label="Track margin")
        ax.plot(xR, yR, "k-", linewidth=1, label="Track margin")

        # Color the road surface
        plt.fill(
            np.concatenate([xL, xR[::-1]]),
            np.concatenate([yL, yR[::-1]]),
            color="grey",
            alpha=0.2,
            label="Road Surface",
        )

        # Plot the kerbs
        if plot_kerbs:
            self._plot_kerbs_2D(ax)

    def plot_track_3D(self, ax: plt.figure, plot_kerbs: bool = False) -> None:
        """Plot the 2D representation of the track.

        Args:
            ax: the axis of the plot.
            plot_kerbs: flag to plot the kerbs.
        """

        # Extract left and right margin points
        x, y, z = self.track.x_mid_line, self.track.y_mid_line, self.track.elevation

        # Plot the mid line
        ax.plot(x, y, z, "k-", linewidth=0.5, label="Mid line")

        # Extract left and right margin points
        xL, yL, zL = (
            self.track.x_margin_no_kerb_L,
            self.track.y_margin_no_kerb_L,
            self.track.z_margin_no_kerb_L,
        )
        xR, yR, zR = (
            self.track.x_margin_no_kerb_R,
            self.track.y_margin_no_kerb_R,
            self.track.z_margin_no_kerb_R,
        )

        # Plot track margins
        ax.plot(xL, yL, zL, "k-", linewidth=1, label="Track margin")
        ax.plot(xR, yR, zR, "k-", linewidth=1, label="Track margin")

        # Create road surface polygons
        verts = [
            list(
                zip(
                    np.concatenate([xL, xR[::-1]]),
                    np.concatenate([yL, yR[::-1]]),
                    np.concatenate([zL, zR[::-1]]),
                )
            )
        ]

        road_surface = Poly3DCollection(verts, color="grey", alpha=0.5)
        ax.add_collection3d(road_surface)

        # Plot the kerbs
        if plot_kerbs:
            self._plot_kerbs_3D(ax)

    #   ____       _            _                        _   _               _
    #  |  _ \ _ __(_)_   ____ _| |_ ___   _ __ ___   ___| |_| |__   ___   __| |___
    #  | |_) | '__| \ \ / / _` | __/ _ \ | '_ ` _ \ / _ \ __| '_ \ / _ \ / _` / __|
    #  |  __/| |  | |\ V / (_| | ||  __/ | | | | | |  __/ |_| | | | (_) | (_| \__ \
    #  |_|   |_|  |_| \_/ \__,_|\__\___| |_| |_| |_|\___|\__|_| |_|\___/ \__,_|___/

    def _read_track_data(self, track_data_path: str) -> Track:
        """Read the track data from the file.

        Args:
            track_data_path: path of the racetrack data file.

        Returns:
            Track: racetrack data.
        """

        # Read the track data, and save track data as a Track object
        track = Track(pd.read_csv(track_data_path, sep="\t", comment="#"))

        # Calculate the margins of the track
        (
            track.x_margin_no_kerb_R,
            track.y_margin_no_kerb_R,
            track.z_margin_no_kerb_R,
            track.x_margin_no_kerb_L,
            track.y_margin_no_kerb_L,
            track.z_margin_no_kerb_L,
            track.x_margin_kerb_R,
            track.y_margin_kerb_R,
            track.z_margin_kerb_R,
            track.x_margin_kerb_L,
            track.y_margin_kerb_L,
            track.z_margin_kerb_L,
        ) = self._track_margins_3D(track)

        return track

    def _track_margins_3D(self, track_data: Track) -> tuple:
        """Calculate the margins of the track in 3D.

        Args:
            track_data: racetrack data.

        Returns:
            tuple: (x,y,z) coordinates of the margins of the track. Both with and without kerbs.
        """

        x_mid_line = track_data.x_mid_line
        y_mid_line = track_data.y_mid_line
        z_mid_line = track_data.elevation

        theta_mid_line = track_data.dir_mid_line
        bank_mid_line = track_data.banking
        slope_mid_line = track_data.slope

        width_R_no_kerbs = track_data.width_no_kerbs_R
        width_L_no_kerbs = track_data.width_no_kerbs_L
        width_R_kerbs = track_data.width_kerbs_R
        width_L_kerbs = track_data.width_kerbs_L

        x_margin_no_kerb_R, y_margin_no_kerb_R, z_margin_no_kerb_R = (
            darboux_to_cartesian(
                x_mid_line,
                y_mid_line,
                z_mid_line,
                theta_mid_line,
                bank_mid_line,
                slope_mid_line,
                -width_R_no_kerbs,
            )
        )
        x_margin_no_kerb_L, y_margin_no_kerb_L, z_margin_no_kerb_L = (
            darboux_to_cartesian(
                x_mid_line,
                y_mid_line,
                z_mid_line,
                theta_mid_line,
                bank_mid_line,
                slope_mid_line,
                width_L_no_kerbs,
            )
        )
        x_margin_kerb_R, y_margin_kerb_R, z_margin_kerb_R = darboux_to_cartesian(
            x_mid_line,
            y_mid_line,
            z_mid_line,
            theta_mid_line,
            bank_mid_line,
            slope_mid_line,
            -width_R_kerbs,
        )
        x_margin_kerb_L, y_margin_kerb_L, z_margin_kerb_L = darboux_to_cartesian(
            x_mid_line,
            y_mid_line,
            z_mid_line,
            theta_mid_line,
            bank_mid_line,
            slope_mid_line,
            width_L_kerbs,
        )

        return (
            x_margin_no_kerb_R,
            y_margin_no_kerb_R,
            z_margin_no_kerb_R,
            x_margin_no_kerb_L,
            y_margin_no_kerb_L,
            z_margin_no_kerb_L,
            x_margin_kerb_R,
            y_margin_kerb_R,
            z_margin_kerb_R,
            x_margin_kerb_L,
            y_margin_kerb_L,
            z_margin_kerb_L,
        )

    def _read_track_origin(self, track_data_path: str) -> tuple:
        """Read the origin of the track.

        Args:
            track_data_path: path of the racetrack data file.

        Returns:
            tuple: (latitude, longitude, altitude) of the origin of the track.
        """

        with open(track_data_path, "r") as file:
            content = file.read()

        finish_line_latitude = re.search(
            r"#! FinishLineLatitude\s*=\s*([-\d.]+)", content
        )
        finish_line_longitude = re.search(
            r"#! FinishLineLongitude\s*=\s*([-\d.]+)", content
        )
        finish_line_altitude = re.search(
            r"#! FinishLineAltitude\s*=\s*([-\d.]+)", content
        )

        origin_x0 = re.search(r"#! x0\s*=\s*([-\d.]+)", content)
        origin_y0 = re.search(r"#! y0\s*=\s*([-\d.]+)", content)
        origin_theta0 = re.search(r"#! theta0\s*=\s*([-\d.]+)", content)

        # Extract and convert the values if found
        if finish_line_latitude:
            finish_line_latitude = float(finish_line_latitude.group(1))
        if finish_line_longitude:
            finish_line_longitude = float(finish_line_longitude.group(1))
        if finish_line_altitude:
            finish_line_altitude = float(finish_line_altitude.group(1))
        if origin_x0:
            origin_x0 = float(origin_x0.group(1))
        if origin_y0:
            origin_y0 = float(origin_y0.group(1))
        if origin_theta0:
            origin_theta0 = float(origin_theta0.group(1))

        # If the origin is not found, return a warning
        if (
            finish_line_latitude,
            finish_line_longitude,
            finish_line_altitude,
            origin_x0,
            origin_y0,
            origin_theta0,
        ) == (
            None,
            None,
            None,
            None,
            None,
            None,
        ):
            print("Warning: origin of the track not found: ", self.track_data_path)

        return (
            finish_line_latitude,
            finish_line_longitude,
            finish_line_altitude,
            origin_x0,
            origin_y0,
            origin_theta0,
        )

    def _plot_kerbs_2D(self, ax: plt.figure, num_stripes: int = 1000) -> None:
        """Plot the kerbs of the track.

        Args:
            num_stripes: number of stripes to plot the kerbs.
        """

        # Extract left and right margin points
        xL, yL = self.track.x_margin_no_kerb_L, self.track.y_margin_no_kerb_L
        xR, yR = self.track.x_margin_no_kerb_R, self.track.y_margin_no_kerb_R
        xLk, yLk = self.track.x_margin_kerb_L, self.track.y_margin_kerb_L
        xRk, yRk = self.track.x_margin_kerb_R, self.track.y_margin_kerb_R

        ax.plot(xLk, yLk, "k-", linewidth=1, label="Track margin")
        ax.plot(xRk, yRk, "k-", linewidth=1, label="Track margin")

        for i in range(num_stripes):

            idx_start = i * len(self.track.x_mid_line) // num_stripes
            idx_end = (i + 1) * len(self.track.x_mid_line) // num_stripes
            color = "red" if i % 2 == 0 else "white"

            # Left kerb
            plt.fill(
                np.concatenate([xL[idx_start:idx_end], xLk[idx_start:idx_end][::-1]]),
                np.concatenate([yL[idx_start:idx_end], yLk[idx_start:idx_end][::-1]]),
                color=color,
                alpha=0.8,
            )

            # Right kerb
            plt.fill(
                np.concatenate([xR[idx_start:idx_end], xRk[idx_start:idx_end][::-1]]),
                np.concatenate([yR[idx_start:idx_end], yRk[idx_start:idx_end][::-1]]),
                color=color,
                alpha=0.8,
            )

    def _plot_kerbs_3D(self, ax: plt.figure, num_stripes: int = 1000) -> None:
        """Plot the kerbs of the track.

        Args:
            num_stripes: number of stripes to plot the kerbs.
        """

        # Extract left and right margin points
        xL, yL, zL = (
            self.track.x_margin_no_kerb_L,
            self.track.y_margin_no_kerb_L,
            self.track.z_margin_no_kerb_L,
        )
        xR, yR, zR = (
            self.track.x_margin_no_kerb_R,
            self.track.y_margin_no_kerb_R,
            self.track.z_margin_no_kerb_R,
        )
        xLk, yLk, zLk = (
            self.track.x_margin_kerb_L,
            self.track.y_margin_kerb_L,
            self.track.z_margin_kerb_L,
        )
        xRk, yRk, zRk = (
            self.track.x_margin_kerb_R,
            self.track.y_margin_kerb_R,
            self.track.z_margin_kerb_R,
        )

        ax.plot(xLk, yLk, zLk, "k-", linewidth=1, label="Track margin")
        ax.plot(xRk, yRk, zRk, "k-", linewidth=1, label="Track margin")

        for i in range(num_stripes):

            idx_start = i * len(self.track.x_mid_line) // num_stripes
            idx_end = (i + 1) * len(self.track.x_mid_line) // num_stripes
            color = "red" if i % 2 == 0 else "white"

            # Left kerb
            verts = [
                list(
                    zip(
                        np.concatenate(
                            [xL[idx_start:idx_end], xLk[idx_start:idx_end][::-1]]
                        ),
                        np.concatenate(
                            [yL[idx_start:idx_end], yLk[idx_start:idx_end][::-1]]
                        ),
                        np.concatenate(
                            [zL[idx_start:idx_end], zLk[idx_start:idx_end][::-1]]
                        ),
                    )
                )
            ]

            road_surface = Poly3DCollection(verts, color=color, alpha=0.8)
            ax.add_collection3d(road_surface)

            # Right kerb
            verts = [
                list(
                    zip(
                        np.concatenate(
                            [xR[idx_start:idx_end], xRk[idx_start:idx_end][::-1]]
                        ),
                        np.concatenate(
                            [yR[idx_start:idx_end], yRk[idx_start:idx_end][::-1]]
                        ),
                        np.concatenate(
                            [zR[idx_start:idx_end], zRk[idx_start:idx_end][::-1]]
                        ),
                    )
                )
            ]

            road_surface = Poly3DCollection(verts, color=color, alpha=0.8)
            ax.add_collection3d(road_surface)
