import numpy as np


def estimate_theta(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate the angle of a 2D curve.

    :param x: x coordinates of the curve.
    :param y: y coordinates of the curve.
    :return: angle of the curve.
    """

    # First derivatives (central difference)
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Angle formula
    theta = np.arctan2(dy, dx)

    return theta


def estimate_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate the curvature of a 2D curve.

    :param x: x coordinates of the curve.
    :param y: y coordinates of the curve.
    :return: curvature of the curve.
    """

    # First derivatives (central difference)
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Second derivatives (central difference)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** (3 / 2)

    return curvature
