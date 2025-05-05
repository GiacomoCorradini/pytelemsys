# %% Setup
import matplotlib.pyplot as plt
import os

from pytelemsys.pytrack import TrackData

# Get root path
root_path = os.path.abspath(os.path.join(__file__, "../"))

# Initialize the TrackData object
track_data = TrackData(os.path.join(root_path, "racetracks/Silverstone_2D.txt"))

# %% Plot track

fig = plt.figure()
ax = fig.add_subplot(111)
track_data.plot_track_2D(ax, plot_kerbs=True)
ax.axis("equal")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
track_data.plot_track_3D(ax, plot_kerbs=True)
ax.axis("equal")

# %% Show the figure
plt.show()
