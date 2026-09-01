"""Broken-axis plots built from the per-cluster mean spectrum.

``plot_cluster_profiles`` is used by both datasets: one broken-axis row
per cluster (labeled with its majority oil), showing that cluster's mean
spectrum. ``plot_cluster_average_vs_random`` is pure-oils only: it adds,
per cluster, one randomly sampled member spectrum plotted alongside the
cluster average, as a sanity check that the average is representative
rather than an artifact of averaging away real spectral variation.

The plotted values are column-wise Z-scores, not raw intensity counts
(see ``clustering/pipeline.py``) - both y-axis labels below say so
explicitly, and negative/zero values are left as-is rather than shifted,
since they are a normal, statistically meaningful part of that
normalization.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..config import LEFT_REGION_MAX, RIGHT_REGION_MIN
from ..plotting.broken_axis import add_break_marks

COUNT_COLUMN = "count_in_each_segment"


def plot_cluster_profiles(
    cluster_profile: pd.DataFrame,
    cluster_colors: dict[str, str],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """One broken-axis row per cluster, plotting its mean spectrum."""
    spectral_data = cluster_profile.drop(columns=[COUNT_COLUMN], errors="ignore")
    wavenumbers = spectral_data.columns.astype(float)
    mask_left = wavenumbers <= LEFT_REGION_MAX
    mask_right = wavenumbers >= RIGHT_REGION_MIN

    fig, axes = plt.subplots(
        len(spectral_data.index), 2, figsize=(14, 10),
        gridspec_kw={"width_ratios": [4, 1]}, sharey=False,
    )

    for row in range(len(spectral_data)):
        # Positional access: two clusters can share the same majority
        # oil label (weak cluster/label alignment on the chips data), so
        # spectral_data.index can contain duplicates - label-based .loc
        # would then return multiple rows instead of this one spectrum.
        cluster = spectral_data.index[row]
        spectrum = spectral_data.iloc[row]
        color = cluster_colors[cluster]

        axes[row, 0].plot(wavenumbers[mask_left], spectrum[mask_left], color=color, linewidth=2)
        axes[row, 1].plot(wavenumbers[mask_right], spectrum[mask_right], color=color, linewidth=2)
        axes[row, 0].set_title(cluster, fontsize=14)
        axes[row, 1].yaxis.tick_right()
        axes[row, 1].tick_params(labelright=False)

        add_break_marks(axes[row, 0], axes[row, 1])

    axes[-1, 0].set_xlabel("Raman Shift (cm$^{-1}$)")
    axes[-1, 1].set_xlabel("Raman Shift (cm$^{-1}$)")
    fig.text(
        0.04, 0.5, "Average Standardized Intensity (z-score)",
        va="center", rotation="vertical", fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_average_vs_random(
    df_with_clusters: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    oil_to_cluster: dict[str, int],
    palette_flat: dict[str, str],
    out_path: Path,
    random_state: int = 42,
    dpi: int = 600,
) -> None:
    """Per oil: cluster-average spectrum vs. one randomly sampled member."""
    spectral_columns = df_with_clusters.columns[:-1]
    wavenumbers = spectral_columns.astype(float)
    mask_left = wavenumbers <= LEFT_REGION_MAX
    mask_right = wavenumbers >= RIGHT_REGION_MIN

    oil_types = cluster_profile.index
    fig = plt.figure(figsize=(12, 10))

    for row, oil in enumerate(oil_types):
        ax_left = plt.subplot(len(oil_types), 2, 2 * row + 1)
        ax_right = plt.subplot(len(oil_types), 2, 2 * row + 2, sharey=ax_left)

        avg_spectrum = cluster_profile.loc[oil].drop(COUNT_COLUMN)
        avg_spectrum.index = avg_spectrum.index.astype(float)

        cluster_id = oil_to_cluster[oil]
        random_row = (
            df_with_clusters[df_with_clusters["K_means_segments"] == cluster_id]
            .sample(1, random_state=random_state)
            .iloc[0, :-1]
        )
        random_row.index = random_row.index.astype(float)

        ax_left.plot(
            avg_spectrum.index[mask_left], avg_spectrum.values[mask_left],
            color=palette_flat[oil], lw=2.5, label="Cluster Average",
        )
        ax_left.plot(
            random_row.index[mask_left], random_row.values[mask_left],
            "k--", lw=1.2, alpha=0.8, label="Random Sample",
        )
        ax_right.plot(
            avg_spectrum.index[mask_right], avg_spectrum.values[mask_right],
            color=palette_flat[oil], lw=2.5,
        )
        ax_right.plot(
            random_row.index[mask_right], random_row.values[mask_right],
            "k--", lw=1.2, alpha=0.8,
        )

        count = cluster_profile.loc[oil, COUNT_COLUMN]
        ax_left.set_title(f"{oil} (n={count})", fontsize=13, fontweight="bold")
        ax_left.set_xlim(500, 1900)
        ax_right.set_xlim(2600, 3000)
        ax_left.grid(alpha=0.3)
        ax_right.grid(alpha=0.3)
        ax_right.tick_params(labelleft=False)

        add_break_marks(ax_left, ax_right)

        if row == 0:
            ax_left.legend(loc="upper left")

    fig.supxlabel("Raman Shift (cm$^{-1}$)", fontsize=14)
    fig.supylabel("Standardized Intensity (z-score)", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
