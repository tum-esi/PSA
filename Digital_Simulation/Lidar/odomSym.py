import os
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from kiss_icp.datasets.kitti import KITTIOdometryDataset
from kiss_icp.pipeline import OdometryPipeline
from loguru import logger
from kiss_icp_eval import run_sequence
import sys


ONLY_PREVIEW = False

class frontCroppedKITTIDatasetv2(KITTIOdometryDataset):
    """
        We want to create a new dataset where we crop the point cloud to specific azimuth angles in order to simulate different LiDAR FOVs.
        However, the GT from the original dataset is not valid anymore. And for this reason we need to have a clean run that will set the compariosn meter
        
    """
    
    def __init__(self, *args, azimuth_deg=(-45.0, 45.0), yaw_offset_deg=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._Rz = None
        self._Rz_h = None
        self._Rz_h = None
        
        if azimuth_deg is None:
            self.azimuth_limits = None
        else:
            low, high = azimuth_deg
            if low > high:
                raise ValueError("azimuth_deg lower bound must be <= upper bound")
            self.azimuth_limits = np.deg2rad((low, high))
            
        """ If we want to apply a rotation over the whole PCD for the visualization of the KITTI preview. """
        if yaw_offset_deg != 0.0: 
            self.yaw_offset = np.deg2rad(yaw_offset_deg)
        
            c, s = np.cos(self.yaw_offset), np.sin(self.yaw_offset)
            self._Rz = np.array([[c, -s, 0.0],
                                [s,  c, 0.0],
                                [0.0, 0.0, 1.0]])
            self._Rz_h = np.eye(4)
            self._Rz_h[:3, :3] = self._Rz
            if getattr(self, "gt_poses", None) is not None:
                self.gt_poses = self._transform_poses(self.gt_poses)

    def _transform_poses(self, poses: np.ndarray) -> np.ndarray:
        rotated = self._Rz_h @ poses
        rotated = rotated @ self._Rz_h.T
        return rotated

    def read_point_cloud(self, scan_file: str):
        points = super().read_point_cloud(scan_file)
        points = points @ self._Rz.T if self._Rz is not None else points   # yaw the whole scan if we applied the transformation
        
        if self.azimuth_limits is None:
            return points
        
        azimuth = np.arctan2(points[:, 1], points[:, 0])
        mask = (azimuth >= self.azimuth_limits[0]) & (azimuth <= self.azimuth_limits[1])
        if not np.any(mask):
            return points
        return points[mask]
    
    pass

def preview_first_scan(
    dataset: frontCroppedKITTIDatasetv2,
    sequence: int,
    label: str,
    max_points: int = 200_000,
):
    """Save a simple XY scatter of the first frame to disk for inspection."""
    points, _ = dataset[0]
    if points.size == 0:
        logger.warning("Sequence {:02d}: first scan empty after cropping", sequence)
        return

    # ---- ROTATE POINT CLOUD SO DRIVING DIRECTION IS UP ----
    # 90° rotation around z (x→y, y→-x)
    R = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]], dtype=np.float32)
    points_rot = points.copy()
    points_rot[:, :3] = points[:, :3] @ R.T
    # --------------------------------------------------------

    full_points = points
    preview_points = points_rot

    if preview_points.shape[0] > max_points:
        choice = np.random.choice(preview_points.shape[0], size=max_points, replace=False)
        preview_points = preview_points[choice]

    preview_dir = Path("plots/previews") / f"seq_{sequence:02d}" / label
    preview_dir.mkdir(parents=True, exist_ok=True)

    pcd_file = preview_dir / "frame000_cropped.pcd"
    write_pcd(full_points, pcd_file)

    # ----- PREVIEW PLOT -----
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(
        preview_points[:, 0],
        preview_points[:, 1],
        color="#000000",
        s=0.08,
        linewidths=0,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim([-20, 20])
    ax.set_ylim([0, 15])
    ax.set_aspect("equal", adjustable="box")

    outfile_png = preview_dir / "frame000.png"
    outfile_pdf = preview_dir / "frame000.pdf"
    fig.savefig(outfile_png, dpi=800, bbox_inches="tight", facecolor="white")
    fig.savefig(outfile_pdf, dpi=800, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Saved first scan preview to {}", outfile_png)
    logger.info("Saved LaTeX-ready PDF preview to {}", outfile_pdf)
    logger.info("Saved cropped point cloud to {}", pcd_file)



def write_pcd(points: np.ndarray, path: Path) -> None:
    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError("points must be an (N,3) or (N,4) array")
    if points.shape[1] == 3:
        zeros = np.zeros((points.shape[0], 1), dtype=points.dtype)
        points = np.hstack((points, zeros))
    header = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z i
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA binary
""".format(n=points.shape[0])
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(points.astype(np.float32).tobytes())

data_root = '/mnt/d' 
kitti_root = os.path.join(data_root, 'data_odometry_velodyne/dataset/')  

if not data_root or not kitti_root:
    logger.error("KITTI path not set. Please set it to the root directory of the KITTI dataset.")
    sys.exit(1)

if not os.path.isdir(kitti_root):
    logger.error("KITTI path does not exist or is not a directory: {}", kitti_root)
    sys.exit(1)

logger.info("Loading dataset @ {}", kitti_root)

FOV_LIDAR = (-75.0, 75.0)
DISPLACEMENT = 20.0
DISPLACEMENT_SWEEPS = [0.0, DISPLACEMENT]


def build_azimuth_runs() -> list[dict]:
    runs = []
    for disp in DISPLACEMENT_SWEEPS:
        if disp == 0.0:
            runs.append(
                {
                    "label": "bening",
                    "azimuth": FOV_LIDAR,
                    "yaw_offset": 0.0,
                    "role": "bening",
                }
            )
            continue
        safe_disp = str(disp).replace("-", "m").replace(".", "p")
        runs.append(
            {
                "label": f"left_{safe_disp}deg",
                "azimuth": (FOV_LIDAR[0] + disp, FOV_LIDAR[1] + disp),
                "yaw_offset": 0.0,
                "role": "variant",
            }
        )
        runs.append(
            {
                "label": f"right_{safe_disp}deg",
                "azimuth": (FOV_LIDAR[0] - disp, FOV_LIDAR[1] - disp),
                "yaw_offset": 0.0,
                "role": "variant",
            }
        )
    return runs


AZIMUTH_RUNS = build_azimuth_runs()



def kitti_sequence(
    sequence: int,
    azimuth=(-45.0, 45.0),
    yaw_offset_deg=0.0,
    preview=False,
    label: str = "gt",
):
    dataset = frontCroppedKITTIDatasetv2(
        data_dir=kitti_root,
        sequence=sequence,
        azimuth_deg=azimuth,
        yaw_offset_deg=yaw_offset_deg,
    )
    if preview:
        preview_first_scan(dataset, sequence, label=label)
    return OdometryPipeline(dataset=dataset)
    
results = {}

"""
    For the first evaluation, we only run the first sequence of the dataset (00)
    The pipeline always requires to have a GT, which we will run first. That will then be used for the runes for comparison. 

"""
logger.info("Starting evaluation runs...")
for seq in range(0, 10):
    for run_cfg in AZIMUTH_RUNS:
        if ONLY_PREVIEW:
            kitti_sequence(
                sequence=seq,
                azimuth=run_cfg["azimuth"],
                yaw_offset_deg=run_cfg["yaw_offset"],
                preview=True,
                label=run_cfg["label"],
            )
            continue
        pipeline_factory = partial(
            kitti_sequence,
            azimuth=run_cfg["azimuth"],
            yaw_offset_deg=run_cfg["yaw_offset"],
            preview=True,
            label=run_cfg["label"],
        )
        if not ONLY_PREVIEW:
            run_sequence(
                pipeline_factory,
                results,
                sequence=seq,
                label=run_cfg["label"],
                role=run_cfg["role"],
            )
