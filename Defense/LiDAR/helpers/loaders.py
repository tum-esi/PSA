import os
import json
from typing import Dict, List, Optional, Tuple, Union

from .pcdHelpers import *
from .ringExtractionHelpers import *


def list_pcd_files(pcd_dir: str) -> List[str]:
    """All .pcd files (sorted for deterministic pairing)."""
    return sorted(
        f for f in os.listdir(pcd_dir) if f.lower().endswith(".pcd")
    )


def load_dir_raw(
    pcd_dir: str,
    metadata_json_path: Optional[str] = None,
    limit: Optional[int] = 300,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all PCDs in a dir.

    Returns dict:
        {
          "file_name.pcd": {
              "refl":  (N,)
              "azi":   (N,)  in degrees
              "theta": (N,)  elevation (rad)
              "ring":  (N,)  int [0..63] or -1
          },
          ...
        }

    Skips unreadable/empty PCDs automatically.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}

    filenames = list_pcd_files(pcd_dir)
    if limit is not None:
        filenames = filenames[:limit]

    for fname in filenames:
        path = os.path.join(pcd_dir, fname)

        try:
            refl2, azi2, theta2 = reflectivity_and_azimuth_from_pcd(path)
            refl = refl2
            azi = azi2
            theta = theta2
            ring = np.full_like(refl, fill_value=-1, dtype=np.int32)
        except Exception as e2:
            print(f"[ERR ] Couldn't even fallback for {fname}: {e2}")
            continue

        out[fname] = {
            "refl": refl,
            "azi": azi,
            "theta": theta,
            
        }
        print(
            f"[INFO] {fname}: {len(refl)} pts "
            f"| refl[{float(np.min(refl)):.3f},{float(np.max(refl)):.3f}]"
        )

    return out