import os
from typing import Dict, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from numbers import Number


# =====================================================================
# ---------------------- Plot / TikZ helpers --------------------------
# =====================================================================

def export_tikz_cartesian(
    out_path: str,
    title: str,
    ylabel: str,
    roi: Tuple[Number, Number],
    profiles: Dict[str, np.ndarray],
    bin_centers: np.ndarray,
    data_header: str,
    y_column_name: str,
):
    """
    Generic TikZ/PGFPlots exporter for Cartesian (x,y) line plots.
    """
    tikz_dir = out_path + "_tikz"
    os.makedirs(tikz_dir, exist_ok=True)

    # Data files
    for label, prof in profiles.items():
        if prof.size == 0:
            continue
        dat_path = os.path.join(tikz_dir, f"{label}.dat")
        with open(dat_path, "w") as f:
            f.write(data_header + "\n")
            for x, y in zip(bin_centers, prof):
                f.write(f"{x:.6f} {y:.6f}\n")

    # PGFPlots file
    tex_path = os.path.join(tikz_dir, "plot.tex")
    with open(tex_path, "w") as f:
        f.write(
            r"""\begin{tikzpicture}
\begin{axis}[
    title={""" + title + r"""},
    xlabel={Azimuth Angle (deg)},
    ylabel={""" + ylabel + r"""},
    xmin=""" + str(roi[0]) + r""",
    xmax=""" + str(roi[1]) + r""",
    grid=both,
]
"""
        )
        for label in profiles:
            f.write(
                "  \\addplot table [x=x_deg, y="
                + y_column_name
                + "] {"
                + f"{label}.dat"
                + "}; % "
                + label
                + "\n"
            )

        f.write(r"\end{axis}" "\n" r"\end{tikzpicture}" "\n")


def plot_line(
    bin_centers: np.ndarray,
    profiles: Dict[str, np.ndarray],
    roi: Tuple[Number, Number],
    title: str,
    out_path: Optional[str] = None,
    show: bool = False,
    export_tikz: bool = True,
):
    """Line plot of multiple profiles with optional TikZ/PGFPlots export."""
    plt.figure(figsize=(12, 6))

    max_y = 0.0
    for label, prof in profiles.items():
        if prof.size == 0:
            continue
        plt.plot(bin_centers, prof, linewidth=2, label=label)
        max_y = max(max_y, float(np.max(prof)) if prof.size else 0.0)

    plt.xlabel("Azimuth Angle (deg)")
    plt.ylabel("Average Reflectivity")
    plt.title(title)
    plt.xlim(roi[0], roi[1])
    plt.ylim(0, max_y * 1.1 if max_y > 0 else 1.0)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

        if export_tikz:
            export_tikz_cartesian(
                out_path,
                title=title,
                ylabel="Average Reflectivity",
                roi=roi,
                profiles=profiles,
                bin_centers=bin_centers,
                data_header="# x_deg  y_value",
                y_column_name="y_value",
            )

    if show:
        plt.show()

    plt.close()


def plot_polar(
    bin_centers: np.ndarray,
    profiles: Dict[str, np.ndarray],
    title: str,
    out_path: Optional[str] = None,
    show: bool = False,
    export_tikz: bool = True,
):
    """Polar plot of multiple profiles + TikZ/PGFPlots export."""
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection="polar")

    max_r = 0.0
    azimuth_rad = np.radians(bin_centers)

    for label, prof in profiles.items():
        if prof.size == 0:
            continue
        ax.plot(azimuth_rad, prof, linewidth=2, label=label)
        max_r = max(max_r, float(np.max(prof)) if prof.size else 0.0)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_rmax(max_r if max_r > 0 else 1.0)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.15))

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

        if export_tikz:
            tikz_dir = out_path + "_tikz"
            os.makedirs(tikz_dir, exist_ok=True)

            # Data files
            for label, prof in profiles.items():
                if prof.size == 0:
                    continue
                dat_path = os.path.join(tikz_dir, f"{label}.dat")
                with open(dat_path, "w") as f:
                    f.write("# angle_deg radius\n")
                    for ang, r in zip(bin_centers, prof):
                        f.write(f"{ang:.6f} {r:.6f}\n")

            # PGFPlots file
            tikz_path = os.path.join(tikz_dir, "plot.tex")
            with open(tikz_path, "w") as f:
                f.write(
                    r"""\begin{tikzpicture}
\begin{polaraxis}[
    title={""" + title + r"""},
    grid=both,
]
"""
                )
                for label in profiles:
                    f.write(
                        r"  \addplot table [x=angle_deg, y=radius] {"
                        + f"{label}.dat"
                        + "}; % "
                        + label
                        + "\n"
                    )
                f.write(r"\end{polaraxis}" "\n" r"\end{tikzpicture}" "\n")

    if show:
        plt.show()

    plt.close()


def plot_per_ring_grid(
    centers: np.ndarray,
    ring_profiles: Dict[int, np.ndarray],
    roi: Tuple[Number, Number],
    title: str,
    out_path: str,
    show: bool = False,
    grid_rows: int = 8,
    grid_cols: int = 8,
):
    """
    Plot each ring's reflectivity-vs-azimuth as its own small subplot
    in a (grid_rows x grid_cols) grid.
    """
    ring_ids = sorted(ring_profiles.keys())
    num_rings = len(ring_ids)

    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(grid_cols * 2.0, grid_rows * 2.0),
        sharex=True,
        sharey=True,
    )

    axes = np.atleast_2d(axes)

    for idx, ring_id in enumerate(ring_ids):
        if idx >= grid_rows * grid_cols:
            break

        r = idx // grid_cols
        c = idx % grid_cols
        ax = axes[r, c]

        prof = ring_profiles[ring_id]
        ax.plot(centers, prof, linewidth=1.0)

        ax.set_title(f"ring {ring_id:02d}", fontsize=8)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.tick_params(axis="both", which="both", labelsize=6, length=2)

    total_cells = grid_rows * grid_cols
    for idx in range(num_rings, total_cells):
        r = idx // grid_cols
        c = idx % grid_cols
        axes[r, c].axis("off")

    fig.suptitle(
        f"{title}\nROI {roi[0]}° .. {roi[1]}°",
        fontsize=12,
        y=0.92,
    )

    plt.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.86,
        bottom=0.07,
        wspace=0.3,
        hspace=0.4,
    )

    axes[grid_rows - 1, 0].set_xlabel("Azimuth (deg)", fontsize=8)
    axes[grid_rows - 1, 0].set_ylabel("Reflectivity", fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    
    
