# %% Setup
import os
import matplotlib.pyplot as plt

from pytelemsys.pyfastf1 import TelemetryFastF1

# Get root path
ROOTH_PATH = os.path.dirname(__file__)

# %% Load formula 1 telemetry data
telemetry = TelemetryFastF1(2025, "Brazil", "R")

leclerc = telemetry.get_driver("LEC")
piastri = telemetry.get_driver("PIA")
antonelli = telemetry.get_driver("ANT")

# %% Selected laps
selected_laps = []
lap_numbers = [5, 6]
for driver in (leclerc, piastri, antonelli):
    tmp_lap = TelemetryFastF1.select_laps(driver, lap_numbers)
    selected_laps.append({"driver": driver["driver"], "lap": TelemetryFastF1.get_data(tmp_lap["data"])})

# %% Plots
lower_bound = 4000  # meters
upper_bound = 6000  # meters
fig, ax = plt.subplots(5, 1, figsize=(10, 10))
for lap_info in selected_laps:
    driver_name = lap_info["driver"]
    lap_df = lap_info["lap"]
    seg = lap_df.loc[(lap_df["Distance"] >= lower_bound) & (lap_df["Distance"] <= upper_bound)]

    ax[0].plot(
        seg["Distance"],
        seg["Speed"],
        label=driver_name,
    )
ax[0].set_ylabel("Speed [km/h]")
ax[0].grid()

for lap_info in selected_laps:
    driver_name = lap_info["driver"]
    lap_df = lap_info["lap"]
    seg = lap_df.loc[(lap_df["Distance"] >= lower_bound) & (lap_df["Distance"] <= upper_bound)]

    ax[1].plot(
        seg["Distance"],
        seg["RPM"],
        label=driver_name,
    )
ax[1].set_ylabel("RPM")
ax[1].grid()

for lap_info in selected_laps:
    driver_name = lap_info["driver"]
    lap_df = lap_info["lap"]
    seg = lap_df.loc[(lap_df["Distance"] >= lower_bound) & (lap_df["Distance"] <= upper_bound)]

    ax[2].plot(
        seg["Distance"],
        seg["nGear"],
        label=driver_name,
    )
ax[2].set_ylabel("Gear")
ax[2].grid()

for lap_info in selected_laps:
    driver_name = lap_info["driver"]
    lap_df = lap_info["lap"]
    seg = lap_df.loc[(lap_df["Distance"] >= lower_bound) & (lap_df["Distance"] <= upper_bound)]

    ax[3].plot(
        seg["Distance"],
        seg["Throttle"],
        label=driver_name,
    )
ax[3].set_ylabel("Throttle [%]")
ax[3].grid()

for lap_info in selected_laps:
    driver_name = lap_info["driver"]
    lap_df = lap_info["lap"]
    seg = lap_df.loc[(lap_df["Distance"] >= lower_bound) & (lap_df["Distance"] <= upper_bound)]

    ax[4].plot(
        seg["Distance"],
        seg["Brake"],
        label=driver_name,
    )
ax[4].set_xlabel("Distance [m]")
ax[4].set_ylabel("Brake [On/Off]")
ax[4].grid()

# hide x-axis ticks/labels on all but the last subplot
for i in range(len(ax) - 1):
    ax[i].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

# place legend horizontally below the axes
ncol = len(selected_laps) if selected_laps else 1
ax[4].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=ncol)
fig.subplots_adjust(bottom=0.22)

# %%
plt.show()
