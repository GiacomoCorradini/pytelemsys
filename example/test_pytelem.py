# %% Setup
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from pytelemsys import TelemetryData
from pytelemsys import TrackData

from pytelemsys.utils import compute_curvilinear_coordinates

# Get root path
root_path = os.path.abspath(os.path.join(__file__, "../"))

# %% Conversion function
def telem_converter(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the input telemetry DataFrame by renaming columns and calculating derived metrics.

    :param data: A pandas DataFrame containing telemetry data with predefined column names.
    :return: A transformed pandas DataFrame with standardized column names and computed metrics.
    """

    data["V"] = np.sqrt(data["Vx"] ** 2 + data["Vy"] ** 2)

    data = data.rename(
        columns={
            "Vx": "u",
            "Vy": "v",
        }
    )

    return data


# %% Initialize the TrackData object
telem_data = TelemetryData(
    os.path.join(root_path, "telemetry/MLT_PA_2025.csv"),
    separator="\t",
    comment="#",
    decimal=".",
    fun_conversion=telem_converter,
)

# %% Track data
track_data = TrackData(os.path.join(root_path, "racetracks/PistaAzzurra_2D.txt"))

# %% Resample the data
sampling_freq = 100
telem_data_resample = telem_data.resample(ref_column="time", freq=sampling_freq)

# Validate that all time differences are equal to 1/freq
time_diff = telem_data_resample["time"].diff().dropna()
expected_diff = 1 / sampling_freq
if not all(abs(time_diff - expected_diff) < 1e-6):
    raise ValueError(
        "Time differences in resampled data are not consistent with the expected frequency."
    )

# %% Plot the data resampled (check only the accelerations)
fig, ax = plt.subplots(3, 1, figsize=(10, 6))
ax[0].plot(telem_data.data["time"], telem_data.data["ax"], label="Raw")
ax[0].plot(
    telem_data_resample["time"],
    telem_data_resample["ax"],
    "--",
    label="Resampled",
)


ax[1].plot(telem_data.data["time"], telem_data.data["ay"], label="Raw")
ax[1].plot(
    telem_data_resample["time"],
    telem_data_resample["ay"],
    "--",
    label="Resampled",
)

ax[2].plot(telem_data.data["time"], telem_data.data["yaw_rate"], label="Raw")
ax[2].plot(
    telem_data_resample["time"],
    telem_data_resample["yaw_rate"],
    "--",
    label="Resampled",
)

fig.tight_layout(pad=0)

# %% Plot trajectory
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
track_data.plot_track_2D(ax=ax, plot_kerbs=False)
ax.plot(
    telem_data.data["x"],
    telem_data.data["y"],
    label="OCP",
    color="red",
    alpha=0.5,
)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title("Trajectory")
ax.legend()

# %% Compute curvilinear coordinates
telem_data.data["s"], telem_data.data["n"] = compute_curvilinear_coordinates(
    track_data.track,
    telem_data.data["x"],
    telem_data.data["y"],
)

# %%
plt.show()
