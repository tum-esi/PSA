import os
from typing import Optional, Tuple, Dict
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from helpers.plottingHelpers import *
from helpers.loaders import *
from numbers import Number

def _validate_roi(roi: Tuple[Number, Number]) -> Tuple[float, float]:
    """Ensure ROI is valid and return as floats."""
    min_roi, max_roi = float(roi[0]), float(roi[1])
    if max_roi <= min_roi:
        raise ValueError("ROI max must be > ROI min")
    return min_roi, max_roi


def build_bin_edges(
    min_roi: float,
    max_roi: float,
    bin_size_deg: Number,
) -> np.ndarray:
    bin_edges = np.arange(min_roi, max_roi + bin_size_deg, bin_size_deg, dtype=float)
    return bin_edges


def apply_savgol_smoothing(
    values: np.ndarray,
    smooth: bool,
    sg_window: int,
    sg_poly: int,
) -> np.ndarray:
    if not smooth or values.size == 0:
        return values

    # Ensure window is valid, odd, and <= len(values)
    window = min(sg_window, len(values))
    if window % 2 == 0:
        window = max(window - 1, 3)
    if window < 3 or window > len(values):
        return values

    return savgol_filter(values, window, sg_poly)


def binned_average(
    azimuth_deg: np.ndarray,
    reflectivity: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """
    Helper to compute average reflectivity per bin.
    """
    counts, _ = np.histogram(azimuth_deg, bins=bin_edges)
    sums, _ = np.histogram(azimuth_deg, bins=bin_edges, weights=reflectivity)
    return np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=float),
        where=counts != 0,
    )


def average_profile(
    azimuth_deg: np.ndarray,
    reflectivity: np.ndarray,
    roi: Tuple[Number, Number],
    bin_size_deg: Number = 1,
    smooth: bool = True,
    sg_window: int = 11,
    sg_poly: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build average reflectivity vs azimuth within ROI.
    Returns (bin_centers_deg, avg_intensity_per_bin).
    """
    min_roi, max_roi = _validate_roi(roi)

    mask = (azimuth_deg >= min_roi) & (azimuth_deg <= max_roi)
    az_roi = azimuth_deg[mask]
    refl_roi = reflectivity[mask]

    if az_roi.size == 0:
        return np.array([]), np.array([])

    bin_edges = build_bin_edges(min_roi, max_roi, bin_size_deg)
    if bin_edges.size < 2:
        return np.array([]), np.array([])

    avg = binned_average(az_roi, refl_roi, bin_edges)
    avg = apply_savgol_smoothing(avg, smooth, sg_window, sg_poly)

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, avg



# =====================================================================
# ------------------------ Defense -----------------------
# =====================================================================

def compute_1dfft(
    bin_centers: np.ndarray,
    profile: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D FFT magnitude of a reflectivity profile sampled over azimuth.

    Args:
        bin_centers: (N,) azimuth angles in degrees (assumed uniformly spaced).
        profile:     (N,) reflectivity values corresponding to bin_centers.

    Returns:
        freqs: (N_fft,) spatial frequencies in cycles/degree (>= 0).
        mag:   (N_fft,) magnitude of the FFT at those frequencies.
    """
    if profile.size == 0 or bin_centers.size < 2:
        return np.array([]), np.array([])

    # Ensure 1D and finite
    profile = np.asarray(profile).ravel()
    bin_centers = np.asarray(bin_centers).ravel()

    # Replace NaNs/Infs with 0 for stability
    profile = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)

    # Estimate sampling step in degrees (assume roughly uniform)
    dtheta = float(np.mean(np.diff(bin_centers)))

    if dtheta <= 0:
        # Fallback: assume 1 degree step
        dtheta = 1.0

    # Frequencies in cycles per degree
    freqs = np.fft.rfftfreq(profile.size, d=dtheta)
    fft_vals = np.fft.rfft(profile)

    mag = np.abs(fft_vals)

    return freqs, mag




def plot_1dfft(
    bin_centers: np.ndarray,
    profiles: Dict[str, np.ndarray],
    roi: Tuple[Number, Number],
    title: str,
    out_path: Optional[str] = None,
    show: bool = False,
    export_tikz: bool = True,
):
    """
    Plot 1D FFT magnitude of reflectivity profiles vs spatial frequency.

    Args:
        bin_centers: (N,) azimuth angles in degrees.
        profiles:    {label: (N,) reflectivity profile}.
        roi:         UNUSED for FFT (kept only for API compatibility).
        title:       Plot title.
        out_path:    Path to save PNG (and TikZ data).
        show:        If True, display the plot interactively.
        export_tikz: If True, export data for TikZ using export_tikz_cartesian.
    """
    plt.figure(figsize=(12, 6))

    fft_spectra: Dict[str, np.ndarray] = {}
    freq_axis: Optional[np.ndarray] = None

    for label, prof in profiles.items():
        if prof.size == 0:
            continue

        freqs, mag = compute_1dfft(bin_centers, prof)
        if freqs.size == 0:
            continue

        if freq_axis is None:
            freq_axis = freqs
        else:
            # In normal use they should all be identical; if not, skip mismatch.
            if freqs.shape != freq_axis.shape or not np.allclose(freqs, freq_axis):
                print(f"[WARN] Skipping {label}: FFT frequency axis mismatch.")
                continue

        fft_spectra[label] = mag
        plt.plot(freqs, mag, linewidth=2, label=label)

    if freq_axis is None:
        print("[WARN] No valid FFT data to plot.")
        plt.close()
        return

    plt.xlabel("Spatial frequency (cycles / degree)")
    plt.ylabel("FFT magnitude")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

        if export_tikz:
            # Use the frequency axis as x for TikZ export
            export_tikz_cartesian(
                out_path,
                title=title,
                ylabel="FFT magnitude",
                roi=(float(freq_axis[0]), float(freq_axis[-1])),
                profiles=fft_spectra,
                bin_centers=freq_axis,
                data_header="# f_cyc_per_deg  |FFT|",
                y_column_name="FFT_mag",
            )

    if show:
        plt.show()

    plt.close()

def detect_optical_lpf_from_fft(
    freqs: np.ndarray,
    mag: np.ndarray,
    f_cut: float = 0.1,
    tol_rel: float = 1e-3,
    tol_abs: float = 1e-2,
) -> bool:
    """
    Scene-independent check for presence of an optical LPF.

    Idea:
      If the LPF is active, all high-frequency magnitudes (freq >= f_cut)
      are ~zero compared to low frequencies.

    Returns:
      True  -> LPF detected (high freqs suppressed)
      False -> No LPF (high freqs still present)
    """
    if freqs.size == 0 or mag.size == 0:
        return False

    # Exclude DC from reference
    low_mask = (freqs > 0) & (freqs < f_cut)
    hf_mask = freqs >= f_cut

    if not np.any(low_mask) or not np.any(hf_mask):
        return False

    low_mag = mag[low_mask]
    hf_mag = mag[hf_mask]

    ref_level = np.max(low_mag)
    threshold = max(tol_abs, tol_rel * ref_level)

    return np.max(hf_mag) < threshold

def high_frequency_energy_ratio(
    freqs: np.ndarray,
    mag: np.ndarray,
    f_cut: float = 0.1,
) -> float:
    if freqs.size == 0 or mag.size == 0:
        return 0.0

    hf_mask = freqs >= f_cut

    total_power = np.sum(mag**2)
    hf_power = np.sum(mag[hf_mask]**2)

    return hf_power / (total_power + 1e-12)

def implementation(
    pcd_dir: str,
    roi: Tuple[Number, Number] = (-180, 180),
    bin_size_deg: float = 1.0,
    f_cut: float = 0.1,
    hf_ratio_thresh: float = 1e-1, 
):

    data = load_dir_raw(pcd_dir)
    if not data:
        print("No data found.")
        return {}

    results = {}

    for fname, d in data.items():
        centers, prof = average_profile(
            d["azi"],
            d["refl"],
            roi,
            bin_size_deg=bin_size_deg,
            smooth=False,
        )

        if prof.size == 0:
            print(f"[SKIP] {fname} — empty profile")
            continue

        # Remove DC for spectral analysis
        prof = prof - np.mean(prof)

        freqs, mag = compute_1dfft(centers, prof)

        # Compute HF energy ratio
        hf_ratio = high_frequency_energy_ratio(freqs, mag, f_cut=f_cut)

        
        # If HF energy is too low → LPF is suppressing high frequencies
        lpf_detected = hf_ratio < hf_ratio_thresh

        results[fname] = {
            "LPF_detected": bool(lpf_detected),
            "HF_energy_ratio": float(hf_ratio),
        }

        status = "PSA Detected (WARNING)" if lpf_detected else "NO PSA"
        print(
            f"{fname}: {status} | HF ratio = {hf_ratio:.6e}"
        )

    return results


def main():
    # Example usage
    results = implementation(
        pcd_dir="<PlaceYourDataHere>",
        roi=(-180, 180),
        bin_size_deg=1.0,
        f_cut=0.1,
        hf_ratio_thresh=1e-1,
    )



if __name__ == "__main__":
    main()