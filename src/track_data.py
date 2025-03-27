import pandas as pd
import matplotlib.pyplot as plt
import re

from src.conversion import *

class TrackData:
    """
    TrackData class stores the track data and the origin of the track.
    """

    def __init__(self, track_data_path:str) -> None:
        """ Initialize the TrackData class.

        Args:
            track_data_path: the path of the track data file.
        """

        # Required fields
        self.required_fields = [
            'abscissa', 'curvature', 'dir_mid_line', 'x_mid_line', 'y_mid_line',
            'elevation', 'slope', 'banking', 'torsion', 'upsilon',
            'width_no_kerbs_L', 'width_no_kerbs_R', 
            'width_kerbs_L', 'width_kerbs_R',
        ]

        # Save the path of the track data
        self.track_data_path = track_data_path

        # Read the origin of the track
        self.origin = self._read_track_origin(track_data_path)

        # Read the track data
        self.track_data = self._read_track_data(track_data_path)

    def plot_track_2D(self, ax:plt.figure, plot_kerbs:bool = False) -> None:
        """ Plot the 2D representation of the track.

        Args:
            ax: the axis of the plot.
            plot_kerbs: flag to plot the kerbs.
        """

        # Plot the mid line
        ax.plot(self.track_data['x_mid_line'], self.track_data['y_mid_line'], 'k-', linewidth=0.5, label='Mid line')

        # Plot the track margins
        ax.plot(self.track_data['x_margin_no_kerb_L'], self.track_data['y_margin_no_kerb_L'], 'k-', linewidth=1, label='Track margin')
        ax.plot(self.track_data['x_margin_no_kerb_R'], self.track_data['y_margin_no_kerb_R'], 'k-', linewidth=1, label='Track margin')

        # Color the road surface
        plt.fill(np.concatenate([self.track_data['x_margin_no_kerb_L'], self.track_data['x_margin_no_kerb_R'][::-1]]), 
                 np.concatenate([self.track_data['y_margin_no_kerb_L'], self.track_data['y_margin_no_kerb_R'][::-1]]), 
                 color='grey', alpha=0.2, label="Road Surface")

        # Plot the kerbs
        if plot_kerbs:
            ax.plot(self.track_data['x_margin_kerb_L'], self.track_data['y_margin_kerb_L'], 'k-', linewidth=1, label='Track margin')
            ax.plot(self.track_data['x_margin_kerb_R'], self.track_data['y_margin_kerb_R'], 'k-', linewidth=1, label='Track margin')
            
            self.plot_kerbs()

    def plot_track_3D(self, ax:plt.figure, plot_kerbs:bool = False) -> None:
        """ Plot the 3D representation of the track.

        Args:
            ax: the axis of the plot.
            plot_kerbs: flag to plot the kerbs.
        """

        # Plot the mid line
        ax.plot(self.track_data['x_mid_line'], self.track_data['y_mid_line'], self.track_data['elevation'], 'k-', linewidth=0.5, label='Mid line')

        # Plot the track margins
        ax.plot(self.track_data['x_margin_no_kerb_L'], self.track_data['y_margin_no_kerb_L'], self.track_data['z_margin_no_kerb_L'], 'k-', linewidth=1, label='Track margin')
        ax.plot(self.track_data['x_margin_no_kerb_R'], self.track_data['y_margin_no_kerb_R'], self.track_data['z_margin_no_kerb_R'], 'k-', linewidth=1, label='Track margin')

        # Plot the kerbs
        if plot_kerbs:
            ax.plot(self.track_data['x_margin_kerb_L'], self.track_data['y_margin_kerb_L'], self.track_data['z_margin_kerb_L'], 'k-', linewidth=1, label='Track margin')
            ax.plot(self.track_data['x_margin_kerb_R'], self.track_data['y_margin_kerb_R'], self.track_data['z_margin_kerb_R'], 'k-', linewidth=1, label='Track margin')
            
    def plot_kerbs(self, num_stripes:int = 1000) -> None:
        """ Plot the kerbs of the track.

        Args:
            num_stripes: number of stripes to plot the kerbs.
        """

        for i in range(num_stripes):

            idx_start = i * len(self.track_data['x_mid_line']) // num_stripes
            idx_end = (i + 1) * len(self.track_data['x_mid_line']) // num_stripes
            color = "red" if i % 2 == 0 else "white"

            # Left kerb
            plt.fill(np.concatenate([self.track_data['x_margin_no_kerb_L'][idx_start:idx_end], self.track_data['x_margin_kerb_L'][idx_start:idx_end][::-1]]),
                     np.concatenate([self.track_data['y_margin_no_kerb_L'][idx_start:idx_end], self.track_data['y_margin_kerb_L'][idx_start:idx_end][::-1]]),
                     color=color, alpha=0.8)

            # Right kerb
            plt.fill(np.concatenate([self.track_data['x_margin_no_kerb_R'][idx_start:idx_end], self.track_data['x_margin_kerb_R'][idx_start:idx_end][::-1]]),
                     np.concatenate([self.track_data['y_margin_no_kerb_R'][idx_start:idx_end], self.track_data['y_margin_kerb_R'][idx_start:idx_end][::-1]]),
                     color=color, alpha=0.8)
            
    #   ____       _            _                        _   _               _     
    #  |  _ \ _ __(_)_   ____ _| |_ ___   _ __ ___   ___| |_| |__   ___   __| |___ 
    #  | |_) | '__| \ \ / / _` | __/ _ \ | '_ ` _ \ / _ \ __| '_ \ / _ \ / _` / __|
    #  |  __/| |  | |\ V / (_| | ||  __/ | | | | | |  __/ |_| | | | (_) | (_| \__ \
    #  |_|   |_|  |_| \_/ \__,_|\__\___| |_| |_| |_|\___|\__|_| |_|\___/ \__,_|___/
                                                                                
    def _check_track_field(self, track_data:pd.DataFrame) -> None:
        """ Check if the required fields are present in the track data.

        Args:
            track_data: racetrack data.

        Raises:
            ValueError: if the required fields are missing in the track data.
        """

        #  Read the track fields
        track_columns = track_data.columns

        # Check if the required fields are present
        if not(set(self.required_fields).issubset(set(track_columns))):

            # Find the missing fields
            missing_fields = list(set(self.required_fields) - set(track_columns))

            raise ValueError(f"Required fields are missing in the track data: {', '.join(missing_fields)}")

    def _read_track_data(self, track_data_path:str) -> pd.DataFrame:
        """ Read the track data from the file.

        Args:
            track_data_path: path of the racetrack data file.

        Returns:
            pd.DataFrame: racetrack data.
        """

        # Read the track data, ignore the lines starting with #
        track_data = pd.read_csv(track_data_path, sep='\t', comment='#')

        # Check if the required fields are present
        self._check_track_field(track_data)

        # Calculate the margins of the track
        track_data['x_margin_no_kerb_R'], \
        track_data['y_margin_no_kerb_R'], \
        track_data['z_margin_no_kerb_R'], \
        track_data['x_margin_no_kerb_L'], \
        track_data['y_margin_no_kerb_L'], \
        track_data['z_margin_no_kerb_L'], \
        track_data['x_margin_kerb_R'], \
        track_data['y_margin_kerb_R'], \
        track_data['z_margin_kerb_R'], \
        track_data['x_margin_kerb_L'], \
        track_data['y_margin_kerb_L'], \
        track_data['z_margin_kerb_L'] \
            = self._track_margins_3D(track_data)

        return track_data

    def _track_margins_3D(self, track_data:pd.DataFrame) -> tuple:
        """ Calculate the margins of the track in 3D.

        Args:
            track_data: racetrack data.

        Returns:
            tuple: (x,y,z) coordinates of the margins of the track. Both with and without kerbs.
        """

        x_mid_line = track_data['x_mid_line']
        y_mid_line = track_data['y_mid_line']
        z_mid_line = track_data['elevation']

        theta_mid_line = track_data['dir_mid_line']
        bank_mid_line = track_data['banking']
        slope_mid_line = track_data['slope']

        width_R_no_kerbs = track_data['width_no_kerbs_R']
        width_L_no_kerbs = track_data['width_no_kerbs_L']
        width_R_kerbs = track_data['width_kerbs_R']
        width_L_kerbs = track_data['width_kerbs_L']

        x_margin_no_kerb_R, y_margin_no_kerb_R, z_margin_no_kerb_R  = darboux_to_cartesian(x_mid_line, y_mid_line, z_mid_line, theta_mid_line, bank_mid_line, slope_mid_line, -width_R_no_kerbs)
        x_margin_no_kerb_L, y_margin_no_kerb_L, z_margin_no_kerb_L  = darboux_to_cartesian(x_mid_line, y_mid_line, z_mid_line, theta_mid_line, bank_mid_line, slope_mid_line, width_L_no_kerbs)
        x_margin_kerb_R, y_margin_kerb_R, z_margin_kerb_R  = darboux_to_cartesian(x_mid_line, y_mid_line, z_mid_line, theta_mid_line, bank_mid_line, slope_mid_line, -width_R_kerbs)
        x_margin_kerb_L, y_margin_kerb_L, z_margin_kerb_L  = darboux_to_cartesian(x_mid_line, y_mid_line, z_mid_line, theta_mid_line, bank_mid_line, slope_mid_line, width_L_kerbs)

        return x_margin_no_kerb_R, y_margin_no_kerb_R, z_margin_no_kerb_R, x_margin_no_kerb_L, y_margin_no_kerb_L, z_margin_no_kerb_L, \
            x_margin_kerb_R, y_margin_kerb_R, z_margin_kerb_R, x_margin_kerb_L, y_margin_kerb_L, z_margin_kerb_L

    def _read_track_origin(self, track_data_path:str) -> tuple:
        """ Read the origin of the track.

        Args:
            track_data_path: path of the racetrack data file.

        Returns:
            tuple: (latitude, longitude, altitude) of the origin of the track.
        """

        with open(track_data_path, 'r') as file:
            content = file.read()

        # Use regular expressions to find the required values
        finish_line_latitude = re.search(r'#! FinishLineLatitude\s*=\s*([-\d.]+)', content)
        finish_line_longitude = re.search(r'#! FinishLineLongitude\s*=\s*([-\d.]+)', content)
        finish_line_altitude = re.search(r'#! FinishLineAltitude\s*=\s*([-\d.]+)', content)

        # Extract and convert the values if found
        if finish_line_latitude:
            finish_line_latitude = float(finish_line_latitude.group(1))
        if finish_line_longitude:
            finish_line_longitude = float(finish_line_longitude.group(1))
        if finish_line_altitude:
            finish_line_altitude = float(finish_line_altitude.group(1))

        # If the origin is not found, return a warning
        if (finish_line_latitude, finish_line_longitude, finish_line_altitude) == (None, None, None):
            print("Warning: origin of the track not found: ", self.track_data_path)

        return finish_line_latitude, finish_line_longitude, finish_line_altitude