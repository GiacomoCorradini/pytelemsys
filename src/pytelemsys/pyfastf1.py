import numpy as np

from pytelemsys.utils.estimation import estimate_curvature


def _require_fastf1():
    """Import FastF1, raising an error if it is not installed."""
    try:
        import fastf1 as ff1

        return ff1
    except ImportError as e:
        raise RuntimeError(
            "FastF1 support is not installed. "
            'Install with: pip install "pytelemsys[fastf1]"'
        ) from e


class TelemetryFastF1:
    """FastF1 session with helpers to extract the lap telemetry."""

    def __init__(self, year: int, weekend: str, session: str) -> None:
        """Load a FastF1 session.

        :param year: season year.
        :param weekend: weekend name.
        :param session: session name.
        """

        # Ensure FastF1 is available
        ff1 = _require_fastf1()

        # Store inputs
        self.year = year
        self.weekend = weekend
        self.session = session

        # Load session
        self.session = ff1.get_session(self.year, self.weekend, self.session)
        self.session.load()

    def get_driver(self, driver: str, fastest: bool = False) -> dict:
        """Get the laps of a driver.

        :param driver: driver code.
        :param fastest: keep only the fastest lap, defaults to False.
        :return: dictionary with the driver code and its laps.
        """

        # Get lap data
        lap = self.session.laps.pick_drivers(driver)
        if fastest:
            lap = lap.pick_fastest()

        return {
            "driver": driver,
            "lap": lap,
        }

    def select_laps(self, driver_data: dict | str, lap_number: int | list):
        """Select the laps of a driver.

        :param driver_data: output of get_driver, or driver code.
        :param lap_number: lap number or list of lap numbers.
        :return: dictionary with the driver code, lap numbers and selected laps.
        """

        if isinstance(driver_data, str):
            driver_data = self.get_driver(driver_data)

        driver_name = driver_data["driver"]
        lap_df = driver_data["lap"]

        # Ensure lap_number is a list
        if isinstance(lap_number, int):
            lap_number = [lap_number]

        # Select laps
        selected = lap_df.loc[lap_df["LapNumber"].isin(lap_number)].copy()

        return {"driver": driver_name, "lap": lap_number, "data": selected}

    @classmethod
    def get_data(cls, lap_data):
        """Get the telemetry of a lap, with estimated ax and ay.

        :param lap_data: FastF1 lap or laps.
        :return: telemetry data.
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

        # DRS open (1) for the values 10, 12, 14, closed (0) otherwise
        car_telem["DRS"] = (car_telem["DRS"] >= 10).astype(int)

        # Estimate the longitudinal acceleration and velocity
        car_telem["Time_s"] = car_telem["Time"] / np.timedelta64(1, "s")
        car_telem["Speed_ms"] = np.gradient(car_telem["Distance"], car_telem["Time_s"])
        car_telem["ax"] = np.gradient(car_telem["Speed_ms"], car_telem["Time_s"])

        # Compute the trajectory curvature (strong approximation)
        pos_telem["curvature"] = estimate_curvature(pos_telem["X"], pos_telem["Y"])

        # Merge the data
        f1_telem = car_telem.merge_channels(pos_telem)

        # slice again to remove the padding, interpolating the first and last value
        f1_telem = f1_telem.slice_by_lap(lap_data, interpolate_edges=True)

        # compute the lateral acceleration, ISO convention: positive in left turns
        # (strong approximation, curvature estimated from the noisy x,y position)
        f1_telem["ay_approx"] = f1_telem["Speed_ms"] ** 2 * f1_telem["curvature"]

        return f1_telem

    def get_circuit_info(self):
        """Get the circuit information.

        :return: FastF1 circuit information.
        """
        return self.session.get_circuit_info()
