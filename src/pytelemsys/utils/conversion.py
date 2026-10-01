import numpy as np
import Clothoids
from pytelemsys.utils.track import Track


def darboux_to_cartesian(
    x_ref: float | np.ndarray,
    y_ref: float | np.ndarray,
    z_ref: float | np.ndarray,
    theta_ref: float | np.ndarray,
    bank_ref: float | np.ndarray,
    slope_ref: float | np.ndarray,
    n: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Convert Darboux coordinates to Cartesian coordinates.

    x, y, z are ENU (z up); the road frame is Rz(theta) Ry(-slope) Rx(-bank).

    :param x_ref: x coordinate of the reference point.
    :param y_ref: y coordinate of the reference point.
    :param z_ref: z coordinate of the reference point.
    :param theta_ref: heading angle of the reference point.
    :param bank_ref: bank angle of the reference point (positive: right side higher).
    :param slope_ref: slope angle of the reference point (positive uphill).
    :param n: lateral distance from the reference point (positive to the left).
    :return: x, y, z coordinates in Cartesian system.
    """

    s_bank = np.sin(bank_ref)
    c_bank = np.cos(bank_ref)
    s_slope = np.sin(slope_ref)
    c_slope = np.cos(slope_ref)
    s_theta = np.sin(theta_ref)
    c_theta = np.cos(theta_ref)

    x = x_ref + n * (c_theta * s_slope * s_bank - s_theta * c_bank)
    y = y_ref + n * (s_theta * s_slope * s_bank + c_theta * c_bank)
    z = z_ref - n * c_slope * s_bank

    return x, y, z


def gps_to_enu(
    latitude: np.ndarray,
    longitude: np.ndarray,
    altitude: np.ndarray,
    origin: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert GPS coordinates to ENU (East, North, Up) coordinates.

    :param latitude: latitude in deg.
    :param longitude: longitude in deg.
    :param altitude: altitude in m.
    :param origin: origin (latitude, longitude, altitude) in deg and m.
    :return: East, North, Up coordinates in m.
    """

    lat0, lon0 = np.radians(origin[0]), np.radians(origin[1])
    P = _geodetic_to_ecef(np.radians(latitude), np.radians(longitude), altitude)
    P0 = _geodetic_to_ecef(lat0, lon0, origin[2])

    # East, North, Up unit vectors at origin in ECEF
    uvec_E0 = np.array([-np.sin(lon0), np.cos(lon0), 0])
    uvec_N0 = np.array(
        [-np.cos(lon0) * np.sin(lat0), -np.sin(lon0) * np.sin(lat0), np.cos(lat0)]
    )
    uvec_U0 = np.array(
        [np.cos(lon0) * np.cos(lat0), np.sin(lon0) * np.cos(lat0), np.sin(lat0)]
    )

    # Projection on tangent plane
    DP = P - P0
    return np.dot(DP, uvec_E0), np.dot(DP, uvec_N0), np.dot(DP, uvec_U0)


def _geodetic_to_ecef(lat, lon, alt) -> np.ndarray:
    """Convert WGS 84 geodetic coordinates to ECEF.

    :param lat: latitude in rad.
    :param lon: longitude in rad.
    :param alt: altitude in m.
    :return: ECEF coordinates, with shape (..., 3).
    """

    a = 6378137.0  # Semi-major axis
    e2 = 0.0818191908426215**2  # Squared eccentricity

    # Radius of curvature in the prime vertical, at each point's latitude
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)

    return np.stack(
        (
            (N + alt) * np.cos(lat) * np.cos(lon),
            (N + alt) * np.cos(lat) * np.sin(lon),
            ((1 - e2) * N + alt) * np.sin(lat),
        ),
        axis=-1,
    )


def cartesian_to_curvilinear(
    track: Track, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to curvilinear coordinates (s, n).

    :param track: track data.
    :param x: x coordinates of the trajectory.
    :param y: y coordinates of the trajectory.
    :return: curvilinear abscissa s and lateral offset n.
    """
    # Check that x and y have the same length
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    # Compute curvilinear coordinates
    clothoid_track = Clothoids.ClothoidList()

    clothoid_track.build(
        x0=track.x_mid_line[0],
        y0=track.y_mid_line[0],
        theta0=track.dir_mid_line[0],
        s=track.abscissa,
        kappa=track.curvature,
    )

    s, n = zip(*(clothoid_track.findST1(xi, yi) for xi, yi in zip(x, y)))

    return np.array(s), np.array(n)
