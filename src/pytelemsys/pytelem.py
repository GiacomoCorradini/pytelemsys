import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

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
        fun_conversion: callable = None,
    ) -> None:

        # Read telemetry data from file
        telem_data = pd.read_csv(
            telem_data_path, sep=separator, comment=comment, decimal=decimal
        )

        # Raise a warning fi s & n are not in the data
        if "n" not in telem_data or "s" not in telem_data:
            warnings.warn("Missing curvilinear coordinates", UserWarning)

    def resample(
        self,
        ref_column: str = "time",
        freq: str = "100",
    ) -> None:
        """Resample the data."""

        return resample_data(self.data, ref_column=ref_column, freq=freq)

    def compute_curvilinear_coordinates(self):
        pass

    #   ____  _       _      __                  _   _
    #  |  _ \| | ___ | |_   / _|_   _ _ __   ___| |_(_) ___  _ __
    #  | |_) | |/ _ \| __| | |_| | | | '_ \ / __| __| |/ _ \| '_ \
    #  |  __/| | (_) | |_  |  _| |_| | | | | (__| |_| | (_) | | | |
    #  |_|   |_|\___/ \__| |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|

    def plot_trajectory(self, ax: plt.figure) -> None:
        ax.plot(self.data.x, self.data.y, "k-", linewidth=2, label="Trajectory")

    def plot_gg_diagram():
        pass

    def plot_inputs(self, flag_save: bool = False, save_path: str = "") -> None:

        fig = plt.figure("Input comparison")

        ax = fig.add_subplot(211)
        plt.plot(self.data.s, self.data.throttle, "k-")
        ax.set_title("Throttle")
        ax.set_xticklabels([])

        ax = fig.add_subplot(212)
        plt.plot(self.data.s, self.data.brake, "k-")
        ax.set_title("Brake")
        plt.xlabel("s [m]")

        if flag_save:
            fig.savefig(save_path, dpi=300)

    def plot_smart():
        pass
