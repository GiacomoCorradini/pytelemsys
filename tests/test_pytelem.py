# %% Setup
import matplotlib.pyplot as plt
import os

from pytelemsys.pytelem import TelemetryData

# Get root path
root_path = os.path.abspath(os.path.join(__file__, "../"))

# Initialize the TrackData object
track_data = TelemetryData(
    os.path.join(root_path, "telemetry/GP2_OCP.txt"),
    separator="\t",
    comment="#",
    decimal=".",
    fun_conversion=None,
)
