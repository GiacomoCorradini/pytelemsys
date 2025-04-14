# %% Setup
import matplotlib.pyplot as plt
import os

from pytelemsys.pytelem import TelemetryData
from pytelemsys.converter import mlt_gp2_converter, gp2_converter
from pytelemsys.utils.conversion import GPS2XYZ_ENU

# Get root path
root_path = os.path.abspath(os.path.join(__file__, "../"))

# %% Initialize the TrackData object
ocp_telem_data = TelemetryData(
    os.path.join(root_path, "telemetry/GP2_OCP.txt"),
    separator="\t",
    comment="#",
    decimal=".",
    fun_conversion=mlt_gp2_converter,
)

gp2_telem_data = TelemetryData(
    os.path.join(root_path, "telemetry/GP2_TELEM_Q.csv"),
    separator=";",
    comment="#",
    decimal=",",
    fun_conversion=gp2_converter,
)

# %% Convert the ENUs to Cartesian coordinates
gp2_telem_data.data["x"], gp2_telem_data.data["y"], gp2_telem_data.data["z"] = (
    GPS2XYZ_ENU(
        gp2_telem_data.data["lon"],
        gp2_telem_data.data["lat"],
        gp2_telem_data.data["z"],
        origin=(
            gp2_telem_data.data["lon"].iloc[0],
            gp2_telem_data.data["lat"].iloc[0],
            gp2_telem_data.data["z"].iloc[0],
        ),
    )
)

# %% Resample the data
sampling_freq = 100
ocp_telem_data_resample = ocp_telem_data.resample(ref_column="time", freq=sampling_freq)

# Validate that all time differences are equal to 1/freq
time_diff = ocp_telem_data_resample["time"].diff().dropna()
expected_diff = 1 / sampling_freq
if not all(abs(time_diff - expected_diff) < 1e-6):
    raise ValueError(
        "Time differences in resampled data are not consistent with the expected frequency."
    )

# %% Plot the data resampled (check only the accelerations)
fig, ax = plt.subplots(3, 1, figsize=(10, 6))
ax[0].plot(ocp_telem_data.data["time"], ocp_telem_data.data["ax"], label="Raw")
ax[0].plot(
    ocp_telem_data_resample["time"],
    ocp_telem_data_resample["ax"],
    "--",
    label="Resampled",
)


ax[1].plot(ocp_telem_data.data["time"], ocp_telem_data.data["ay"], label="Raw")
ax[1].plot(
    ocp_telem_data_resample["time"],
    ocp_telem_data_resample["ay"],
    "--",
    label="Resampled",
)

ax[2].plot(ocp_telem_data.data["time"], ocp_telem_data.data["yaw_rate"], label="Raw")
ax[2].plot(
    ocp_telem_data_resample["time"],
    ocp_telem_data_resample["yaw_rate"],
    "--",
    label="Resampled",
)

fig.tight_layout(pad=0)

# %% Plot trajectory
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(
    gp2_telem_data.data["x"],
    gp2_telem_data.data["y"],
    label="GP2",
    color="blue",
    alpha=0.5,
)
ax.plot(
    ocp_telem_data.data["x"],
    ocp_telem_data.data["y"],
    label="OCP",
    color="red",
    alpha=0.5,
)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title("Trajectory")
ax.legend()

plt.show()
