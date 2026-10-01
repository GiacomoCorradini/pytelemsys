import numpy as np


def estimate_theta(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate the heading angle of a 2D curve.

    :param x: x coordinates of the curve.
    :param y: y coordinates of the curve.
    :return: heading angle of the curve in rad, unwrapped.
    """

    # First derivatives (central difference)
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Angle formula, unwrapped to avoid 2*pi jumps
    theta = np.unwrap(np.arctan2(dy, dx))

    return theta


def estimate_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate the curvature of a 2D curve.

    :param x: x coordinates of the curve.
    :param y: y coordinates of the curve.
    :return: curvature of the curve in 1/m (positive for left turns).
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
