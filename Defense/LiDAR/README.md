# LiDAR Defense Analysis

Detects optical low-pass filters in LiDAR reflectivity data using 1D FFT spectral analysis.

## Installation
If not done previously:

```bash
pip install -r ../../requirements_lidar.txt
```

## Requirements

The helpers directory must contain:
- `loaders.py` - PCD file loading utilities
- `plottingHelpers.py` - Plotting utilities

## Usage

```python
from Defense import implementation

results = implementation(
    pcd_dir="/path/to/pcd/directory",
    roi=(-180, 180),
    bin_size_deg=1.0,
    f_cut=0.1,
    hf_ratio_thresh=1e-1,
)
```

## Output

Returns a dictionary with LPF detection results:
```python
{
    "filename.pcd": {
        "LPF_detected": bool,
        "HF_energy_ratio": float
    }
}
```
