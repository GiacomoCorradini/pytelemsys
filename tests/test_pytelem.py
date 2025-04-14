# %% Setup
import matplotlib.pyplot as plt
import os
import sys

from pytelemsys.pytelem import TelemetryData
from pytelemsys.converter import mlt_gp2_converter

# Get root path
root_path = os.path.abspath(os.path.join(__file__, "../"))

# %% Initialize the TrackData object
track_data = TelemetryData(
    os.path.join(root_path, "telemetry/GP2_OCP.txt"),
    separator="\t",
    comment="#",
    decimal=".",
    fun_conversion=mlt_gp2_converter,
)

# %% Resample the data
sampling_freq = 100
track_data_resample = track_data.resample(ref_column="time", freq=sampling_freq)

# Validate that all time differences are equal to 1/freq
time_diff = track_data_resample["time"].diff().dropna()
expected_diff = 1 / sampling_freq
if not all(abs(time_diff - expected_diff) < 1e-6):
    raise ValueError(
        "Time differences in resampled data are not consistent with the expected frequency."
    )

# %% Plot the data resampled (check only the accelerations)
fig, ax = plt.subplots(3, 1, figsize=(10, 6))
ax[0].plot(track_data.data["time"], track_data.data["ax"], label="Raw")
ax[0].plot(
    track_data_resample["time"], track_data_resample["ax"], "--", label="Resampled"
)

ax[1].plot(track_data.data["time"], track_data.data["ay"], label="Raw")
ax[1].plot(
    track_data_resample["time"], track_data_resample["ay"], "--", label="Resampled"
)

ax[2].plot(track_data.data["time"], track_data.data["yaw_rate"], label="Raw")
ax[2].plot(
    track_data_resample["time"],
    track_data_resample["yaw_rate"],
    "--",
    label="Resampled",
)

# %%
