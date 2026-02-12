import numpy as np
from numbers import Number
from typing import Tuple


def magnitude(x: Number, y: Number, z: Number) -> Number:
    """||[x,y,z]||. Works with scalars or numpy arrays."""
    return np.sqrt(x * x + y * y + z * z)


def to_spherical(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cartesian -> spherical (r, theta, phi).

    r     : radius
    theta : inclination from +Z (0..pi)
    phi   : azimuth in XY plane (-pi..pi)
    """
    r = magnitude(x, y, z)
    theta = np.arctan2(np.sqrt(x * x + y * y), z)
    phi = np.arctan2(y, x)
    return r, theta, phi

def to_spherical(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cartesian -> spherical (r, theta, phi).

    r     : radius
    theta : inclination from +Z (0..pi)
    phi   : azimuth in XY plane (-pi..pi)
    """
    r = magnitude(x, y, z)
    theta = np.arctan2(np.sqrt(x * x + y * y), z)
    phi = np.arctan2(y, z)
    return r, theta, phi


def normalize_azimuth_deg(phi_rad: np.ndarray) -> np.ndarray:
    """
    Convert radians -> degrees.
    NOTE: docstring says [-180,180] but current implementation just uses degrees,
    preserving original behavior.
    """
    return np.degrees(phi_rad)


def elevation_from_xyz(xyz: np.ndarray) -> np.ndarray:
    """
    Elevation angle = atan2(z, sqrt(x^2 + y^2)) for each point.
    Range ~[-30deg,+15deg] for OS1-64 depending on model.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return np.arctan2(z, np.sqrt(x * x + y * y))


def compute_reflectivity_from_color_or_radius(
    colors: np.ndarray, radius: np.ndarray
) -> np.ndarray:
    """
    Prefer R channel of color if present, otherwise use normalized radius.
    """
    if colors.size > 0:
        return colors[:, 0]

    max_r = float(np.max(radius)) if radius.size > 0 else 0.0
    if max_r > 0.0:
        return radius / max_r
    return np.zeros_like(radius)