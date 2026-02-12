import numpy as np
import json
from typing import Optional
import open3d as o3d

from.geometryHelpers import elevation_from_xyz

# =====================================================================
# -------------------- Ring extraction helpers -----------------------
# =====================================================================

def load_beam_altitudes_rad(metadata_json_path: str) -> np.ndarray:
    """
    Read Ouster metadata JSON and return per-channel vertical angles [rad].
    """
    with open(metadata_json_path, "r") as f:
        meta = json.load(f)
    alts_deg = meta["beam_intrinsics"]["beam_altitude_angles"]
    return np.radians(np.array(alts_deg, dtype=float))


def assign_rings_from_metadata(
    xyz: np.ndarray,
    beam_alts_rad: np.ndarray,
    tol_deg: float = 0.3,
) -> np.ndarray:
    """
    Approximate which laser channel (ring 1..N) produced each 3D point by snapping
    elevation to the closest beam_altitude_angle. Outside tol_deg => ring = -1.
    """
    elev = elevation_from_xyz(xyz)  # (N,) rad
    diffs = elev[:, None] - beam_alts_rad[None, :]  # (N, num_channels)
    abs_diffs = np.abs(diffs)

    idx = np.argmin(abs_diffs, axis=1)
    min_err = np.min(abs_diffs, axis=1)

    ring = idx.astype(np.int32)
    ring[np.degrees(min_err) > tol_deg] = -1
    return ring


def extract_ring_indices_from_pcd(
    pcd_path: str,
    metadata_json_path: Optional[str] = None,
) -> np.ndarray:
    """
    Return an array 'ring' of shape (N,) giving channel index per point.

    Strategy:
    1. Try to read a 'ring' field directly from the PCD using Open3D's tensor API.
    2. If not available, reconstruct using beam_altitude_angles from metadata_json_path.
    """
    tpcd = o3d.t.io.read_point_cloud(pcd_path)

    if "ring" in tpcd.point:
        return tpcd.point["ring"].numpy().astype(np.int32).reshape(-1)

    if metadata_json_path is None:
        raise RuntimeError(
            f"No 'ring' channel present in {pcd_path}, and no metadata_json_path provided."
        )

    xyz = tpcd.point["positions"].numpy()  # (N,3)
    beam_alts_rad = load_beam_altitudes_rad(metadata_json_path)
    return assign_rings_from_metadata(xyz, beam_alts_rad)
