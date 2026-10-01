import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import re
import warnings
import Clothoids

from pytelemsys.utils.track import Track
from pytelemsys.utils import darboux_to_cartesian


class TrackData:
    """Track data and origin read from a track file."""

    def __init__(self, track_data_path: str) -> None:
        """Read the track data and origin from a file.

        :param track_data_path: path of the track file.
        """

        # Save the path of the track data
        self.track_data_path = track_data_path

        # Read the origin of the track
        self.origin_gps, self.origin_pose, self.origin_matrix = self._read_track_origin(
            track_data_path
        )

        # Read the track data
        self.track = self._read_track_data(track_data_path)

    def plot_track_2D(
        self, ax: plt.Axes, plot_kerbs: bool = False, legend: bool = True
    ) -> None:
        """Plot the track in 2D.

        :param ax: matplotlib axes.
        :param plot_kerbs: plot the kerbs, defaults to False.
        :param legend: show the track elements in the legend, defaults to True.
        """

        # Labels starting with "_" are ignored by the legend
        label = lambda name: name if legend else "_" + name

        # Extract left and right margin points
        x, y = self.track.x_mid_line, self.track.y_mid_line

        # Plot the mid line
        ax.plot(x, y, "k-", linewidth=0.5, label=label("Mid line"))

        # Extract left and right margin points
        xL, yL = self.track.x_margin_no_kerb_L, self.track.y_margin_no_kerb_L
        xR, yR = self.track.x_margin_no_kerb_R, self.track.y_margin_no_kerb_R

        # Plot track margins
        ax.plot(xL, yL, "k-", linewidth=1, label=label("Track margin"))
        ax.plot(xR, yR, "k-", linewidth=1)

        # Color the road surface
        ax.fill(
            np.concatenate([xL, xR[::-1]]),
            np.concatenate([yL, yR[::-1]]),
            color="grey",
            alpha=0.2,
            label=label("Road surface"),
        )

        # Plot the kerbs
        if plot_kerbs:
            self._plot_kerbs_2D(ax)

    def plot_track_3D(
        self, ax: plt.Axes, plot_kerbs: bool = False, legend: bool = True
    ) -> None:
        """Plot the track in 3D.

        :param ax: matplotlib 3D axes.
        :param plot_kerbs: plot the kerbs, defaults to False.
        :param legend: show the track elements in the legend, defaults to True.
        """

        # Labels starting with "_" are ignored by the legend
        label = lambda name: name if legend else "_" + name

        # Extract left and right margin points
        x, y, z = self.track.x_mid_line, self.track.y_mid_line, self.track.elevation

        # Plot the mid line
        ax.plot(x, y, z, "k-", linewidth=0.5, label=label("Mid line"))

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
        ax.plot(xL, yL, zL, "k-", linewidth=1, label=label("Track margin"))
        ax.plot(xR, yR, zR, "k-", linewidth=1)

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

    def rebuild_track(
        self,
    ) -> Clothoids.ClothoidList:
        """Build the mid line as a list of clothoids.

        :return: mid line clothoid list.
        """

        # Compute curvilinear coordinates
        clothoid_track = Clothoids.ClothoidList()

        clothoid_track.build(
            x0=self.track.x_mid_line[0],
            y0=self.track.y_mid_line[0],
            theta0=self.track.dir_mid_line[0],
            s=self.track.abscissa,
            kappa=self.track.curvature,
        )

        return clothoid_track

    #   ____       _            _                        _   _               _
    #  |  _ \ _ __(_)_   ____ _| |_ ___   _ __ ___   ___| |_| |__   ___   __| |___
    #  | |_) | '__| \ \ / / _` | __/ _ \ | '_ ` _ \ / _ \ __| '_ \ / _ \ / _` / __|
    #  |  __/| |  | |\ V / (_| | ||  __/ | | | | | |  __/ |_| | | | (_) | (_| \__ \
    #  |_|   |_|  |_| \_/ \__,_|\__\___| |_| |_| |_|\___|\__|_| |_|\___/ \__,_|___/

    def _read_track_data(self, track_data_path: str) -> Track:
        """Read the track data and compute its margins.

        :param track_data_path: path of the track file.
        :return: track data.
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
        """Compute the 3D margins of the track, without and with kerbs.

        :param track_data: track data.
        :return: x, y, z of the right and left margins, without and then with kerbs.
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

    def _read_track_origin(
        self, track_data_path: str
    ) -> tuple[tuple, tuple, np.ndarray]:
        """Read the origin of the track from the file header.

        :param track_data_path: path of the track file.
        :return: GPS origin (latitude, longitude, altitude), mid line start pose
            (x0, y0, z0, theta0) and matrix from the mid line start frame to the
            track frame. Missing values are None.
        """

        # Header lines are "#! key = value"
        header = {}
        with open(track_data_path, "r") as file:
            content = file.read()
        for key, value in re.findall(r"^#!\s*(\w+)\s*=\s*(\S+)", content, re.M):
            try:
                header[key] = float(value)
            except ValueError:
                pass

        gps_keys = ("FinishLineLatitude", "FinishLineLongitude", "FinishLineAltitude")
        gps = tuple(header.get(key) for key in gps_keys)
        pose = tuple(header.get(key) for key in ("x0", "y0", "z0", "theta0"))

        if all(value is None for value in gps + pose):
            warnings.warn(f"Origin of the track not found: {track_data_path}")

        # Roto-translation from the mid line start frame to the track frame
        x0, y0, _, theta0 = pose
        if None in (x0, y0, theta0):
            matrix = np.eye(3)
        else:
            c, s = np.cos(theta0), np.sin(theta0)
            matrix = np.array([[c, -s, x0], [s, c, y0], [0, 0, 1]])

        return gps, pose, matrix

    def _kerb_stripes(self, stripe_length: float) -> list[slice]:
        """Split the track in stripes of about stripe_length.

        :param stripe_length: length of each stripe in m.
        :return: index slices of the stripes, sharing the boundary points.
        """

        s = self.track.abscissa
        bounds = np.searchsorted(s, np.arange(s[0], s[-1], stripe_length))
        bounds = np.unique(np.append(bounds, len(s) - 1))

        return [slice(a, b + 1) for a, b in zip(bounds[:-1], bounds[1:])]

    def _plot_kerbs_2D(self, ax: plt.Axes, stripe_length: float = 5.0) -> None:
        """Plot the kerbs in 2D.

        :param ax: matplotlib axes.
        :param stripe_length: length of the red and white stripes in m, defaults to 5.
        """

        # Extract left and right margin points
        xL, yL = self.track.x_margin_no_kerb_L, self.track.y_margin_no_kerb_L
        xR, yR = self.track.x_margin_no_kerb_R, self.track.y_margin_no_kerb_R
        xLk, yLk = self.track.x_margin_kerb_L, self.track.y_margin_kerb_L
        xRk, yRk = self.track.x_margin_kerb_R, self.track.y_margin_kerb_R

        ax.plot(xLk, yLk, "k-", linewidth=1)
        ax.plot(xRk, yRk, "k-", linewidth=1)

        for i, idx in enumerate(self._kerb_stripes(stripe_length)):
            color = "red" if i % 2 == 0 else "white"

            # Left kerb
            ax.fill(
                np.concatenate([xL[idx], xLk[idx][::-1]]),
                np.concatenate([yL[idx], yLk[idx][::-1]]),
                color=color,
                alpha=0.8,
            )

            # Right kerb
            ax.fill(
                np.concatenate([xR[idx], xRk[idx][::-1]]),
                np.concatenate([yR[idx], yRk[idx][::-1]]),
                color=color,
                alpha=0.8,
            )

    def _plot_kerbs_3D(self, ax: plt.Axes, stripe_length: float = 5.0) -> None:
        """Plot the kerbs in 3D.

        :param ax: matplotlib 3D axes.
        :param stripe_length: length of the red and white stripes in m, defaults to 5.
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

        ax.plot(xLk, yLk, zLk, "k-", linewidth=1)
        ax.plot(xRk, yRk, zRk, "k-", linewidth=1)

        for i, idx in enumerate(self._kerb_stripes(stripe_length)):
            color = "red" if i % 2 == 0 else "white"

            # Left kerb
            verts = zip(
                np.concatenate([xL[idx], xLk[idx][::-1]]),
                np.concatenate([yL[idx], yLk[idx][::-1]]),
                np.concatenate([zL[idx], zLk[idx][::-1]]),
            )
            ax.add_collection3d(Poly3DCollection([list(verts)], color=color, alpha=0.8))

            # Right kerb
            verts = zip(
                np.concatenate([xR[idx], xRk[idx][::-1]]),
                np.concatenate([yR[idx], yRk[idx][::-1]]),
                np.concatenate([zR[idx], zRk[idx][::-1]]),
            )
            ax.add_collection3d(Poly3DCollection([list(verts)], color=color, alpha=0.8))
