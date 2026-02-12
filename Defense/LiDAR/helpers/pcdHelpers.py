from open3d import open3d as o3d
import numpy as np
from typing import Optional, Tuple
from .geometryHelpers import *
from .ringExtractionHelpers import *
import os

def load_pcd(path: str) -> o3d.geometry.PointCloud:
    """
    Read a PCD via the tensor IO (keeps extra channels) then convert to legacy.
    If 'intensity' is present, it is normalized and mapped to 'colors'.
    """
    try:
        tpcd = o3d.t.io.read_point_cloud(path)
    except Exception as exc:
        raise ValueError(f"Failed to read {os.path.basename(path)}: {exc}") from exc

    if "positions" not in tpcd.point:
        raise ValueError(f"{os.path.basename(path)} is missing positions data.")

    num_points = int(tpcd.point["positions"].shape[0])
    if num_points == 0:
        raise ValueError(f"{os.path.basename(path)} has no points.")

    # map intensity -> colors if needed
    if "intensity" in tpcd.point:
        intensities = tpcd.point["intensity"].numpy().reshape(-1)
        min_intensity = float(np.min(intensities))
        max_intensity = float(np.max(intensities))

        if max_intensity > min_intensity:
            norm_intensities = (intensities - min_intensity) / (
                max_intensity - min_intensity
            )
        else:
            norm_intensities = np.zeros_like(intensities)

        colors = np.stack([norm_intensities] * 3, axis=1)
        tpcd.point["colors"] = o3d.core.Tensor(
            colors, dtype=o3d.core.Dtype.Float32
        )

    return tpcd.to_legacy()


def _reflectivity_azimuth_elev_from_pcd(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shared core: compute (reflectivity, azimuth_deg, elevation_rad) for one PCD.
    """
    pcd = load_pcd(path)
    pts = np.asarray(pcd.points)  # (N,3)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    radius, _, phi = to_spherical(x, y, z)
    azimuth_deg = normalize_azimuth_deg(phi)
    elevation = elevation_from_xyz(pts)

    colors = np.asarray(pcd.colors)  # (N,3) or empty
    print("colors shape:", colors.shape)
    refl = compute_reflectivity_from_color_or_radius(colors, radius)

    return (
        refl.astype(np.float32),
        azimuth_deg.astype(np.float32),
        elevation.astype(np.float32),
    )


def reflectivity_and_azimuth_from_pcd(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backwards-compatible wrapper.

    Return (reflectivity, azimuth_deg, elevation) arrays for one PCD.
    If color is present, use the R channel as reflectivity (0..1).
    Otherwise, fall back to normalized radius.
    """
    return _reflectivity_azimuth_elev_from_pcd(path)
