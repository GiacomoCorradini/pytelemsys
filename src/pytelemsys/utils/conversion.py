import numpy as np


def darboux_to_cartesian(
    x_ref: float,
    y_ref: float,
    z_ref: float,
    theta_ref: float,
    bank_ref: float,
    slope_ref: float,
    n: float,
) -> tuple[float, float, float]:
    """Convert from Darboux coordinates to Cartesian coordinates.

    :param x_ref: x coordinate of the reference point.
    :param y_ref: y coordinate of the reference point.
    :param z_ref: z coordinate of the reference point.
    :param theta_ref: angle of the reference point.
    :param bank_ref: bank angle of the reference point.
    :param slope_ref: slope angle of the reference point.
    :param n: distance from the reference point.
    :return: x, y, z coordinates in Cartesian system.
    """

    s_bank = np.sin(bank_ref)
    c_bank = np.cos(bank_ref)
    s_slope = np.sin(slope_ref)
    c_slope = np.cos(slope_ref)
    s_theta = np.sin(theta_ref)
    c_theta = np.cos(theta_ref)

    x = x_ref - n * s_theta * c_bank + c_theta * s_slope * s_bank
    y = y_ref + n * c_theta * c_bank + s_theta * s_slope * s_bank
    z = z_ref + n * c_slope * s_bank

    return x, y, z


def GPS2XYZ_ENU(
    longitude: np.ndarray,
    latitude: np.ndarray,
    altitude: np.ndarray,
    origin: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert GPS coordinates to ENU (East, North, Up) coordinates.

    :param longitude: longitude in degrees
    :param latitude: latitude in degrees
    :param altitude: altitude in meters
    :param origin: origin coordinates (longitude, latitude, altitude) in degrees and meters
    :return: ENU coordinates (East, North, Up)
    """

    # Origin coordinates
    lon_0 = origin[0]
    lat_0 = origin[1]
    alt0 = origin[2]

    # WGS 84 parameters
    a = 6378137.0  # Semi-major axis
    b = 6356752.31424518  # Semi-minor axis
    f_inv = 298.257223563  # Inverse flattening
    e = 0.0818191908426215  # Eccentricity

    # Calculate squared eccentricity
    e2 = e**2

    # Convert to radians
    lat = np.radians(latitude)
    lon = np.radians(longitude)
    lat0 = np.radians(lat_0)
    lon0 = np.radians(lon_0)

    # Radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat0) ** 2)

    # Calculate normalized Cartesian coordinates
    norm_x = np.cos(lat) * np.cos(lon)
    norm_y = np.cos(lat) * np.sin(lon)
    norm_z = np.sin(lat)

    # Cartesian coordinates of point P in ECEF
    PX = norm_x * (N + altitude)
    PY = norm_y * (N + altitude)
    PZ = norm_z * ((1 - e2) * N + altitude)

    # Cartesian coordinates of the origin in ECEF
    PX0 = np.cos(lon0) * np.cos(lat0) * (N + alt0)
    PY0 = np.sin(lon0) * np.cos(lat0) * (N + alt0)
    PZ0 = np.sin(lat0) * ((1 - e2) * N + alt0)

    # East, North, Up unit vectors at origin in ECEF
    uvec_E0 = np.array([-np.sin(lon0), np.cos(lon0), 0])
    uvec_N0 = np.array(
        [-np.cos(lon0) * np.sin(lat0), -np.sin(lon0) * np.sin(lat0), np.cos(lat0)]
    )
    uvec_U0 = np.array(
        [np.cos(lon0) * np.cos(lat0), np.sin(lon0) * np.cos(lat0), np.sin(lat0)]
    )

    # Position vectors
    P = np.stack((PX, PY, PZ), axis=-1)
    origin = np.array([PX0, PY0, PZ0])

    # Projection on tangent plane
    DP = P - origin
    XYZ = np.stack(
        (np.dot(DP, uvec_E0), np.dot(DP, uvec_N0), np.dot(DP, uvec_U0)), axis=-1
    )

    return XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
