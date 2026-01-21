import numpy as np

from pytelemsys.utils.processing import low_pass_filter, moving_average
from pytelemsys.utils.estimation import estimate_curvature

def _require_fastf1():
    try:
        import fastf1 as ff1
        return ff1
    except ImportError as e:
        raise RuntimeError(
            "FastF1 support is not installed. "
            "Install with: pip install -e '.[fastf1]"
        ) from e

class TelemetryFastF1:
    def __init__(self, _year: int, _weekend: str, _session: str) -> None:
        """Initialize the TelemetryFastF1 class.

        :param _year: year of the season
        :param _weekend: weekend name
        :param _session: session name
        """

        # Ensure FastF1 is available
        ff1 = _require_fastf1()

        # Store inputs
        self.year = _year
        self.weekend = _weekend
        self.session = _session

        # Load session
        self.session = ff1.get_session(self.year, self.weekend, self.session)
        self.session.load()

    def get_driver(self, _driver: str, fastest: bool = False) -> dict:
        """Get driver data.

        :param _driver: driver code
        :param fastest: if true pick the fastest lap, defaults to False
        :return: dictionary with driver name and lap data
        """

        # Get lap data
        lap = self.session.laps.pick_drivers(_driver)
        if fastest:
            lap = lap.pick_fastest()

        return {
            "driver": _driver,
            "lap": lap,
        }

    @classmethod
    def select_laps(cls, driver_data: dict | str, lap_number: int | list):
        """Select laps for a given driver.

        :param driver_data: driver data dictionary or driver code
        :param lap_number: lap number or list of lap numbers
        :return: dictionary with driver name and selected lap data
        """

        if isinstance(driver_data, str):
            driver_data = cls.get_driver(driver_data)

        driver_name = driver_data["driver"]
        lap_df = driver_data["lap"]

        # Ensure lap_number is a list
        if isinstance(lap_number, int):
            lap_number = [lap_number]

        # Select laps
        mask = lap_df["LapNumber"] == lap_number[0]
        for ln in lap_number[1:]:
            mask |= lap_df["LapNumber"] == ln

        selected = lap_df.loc[mask].copy()

        return {"driver": driver_name, "lap": lap_number, "data": selected}

    @classmethod
    def get_data(cls, lap_data):
        """Get telemetry data for a given lap.

        :param lap_data: lap data
        :return: telemetry data
        """
        # Get car and position telemetry
        car_telem = lap_data.get_car_data(pad=1, pad_side="both").add_distance()
        pos_telem = lap_data.get_pos_data(pad=1, pad_side="both")

        # Register new channels (needed for correct merging)
        car_telem.register_new_channel("Time_s", "continuous", "linear")
        car_telem.register_new_channel("Speed_ms", "continuous", "linear")
        car_telem.register_new_channel("ax", "continuous", "linear")
        car_telem.register_new_channel("ay_approx", "continuous", "linear")

        pos_telem.register_new_channel("curvature", "continuous", "linear")

        # Change x,y,z measurements units to meters
        pos_telem["X"] = pos_telem["X"] / 10
        pos_telem["Y"] = pos_telem["Y"] / 10
        pos_telem["Z"] = pos_telem["Z"] / 10

        # Fix the DRS data
        car_telem["DRS"] = [0 if x == 8 else 1 for x in car_telem["DRS"]]

        # Estimate the longitudinal acceleration and velocity
        car_telem["Time_s"] = car_telem["Time"] / np.timedelta64(1, "s")
        car_telem["Speed_ms"] = np.gradient(car_telem["Distance"], car_telem["Time_s"])
        car_telem["ax"] = np.gradient(car_telem["Speed_ms"], car_telem["Time_s"])

        # Compute the trajectory curvature (strong approximation)
        pos_telem["curvature"] = estimate_curvature(pos_telem["X"], pos_telem["Y"])

        # Merge the data
        f1_telem = car_telem.merge_channels(pos_telem)

        # slice again to remove the padding and interpolate the exact first and last value
        f1_telem = f1_telem.slice_by_lap(lap_data, interpolate_edges=True)

        # compute the lateral acceleration (extreme approximation since x,y are the centerline of the track)
        f1_telem["ay_approx"] = -f1_telem["Speed_ms"] ** 2 * f1_telem["curvature"]

        return f1_telem
