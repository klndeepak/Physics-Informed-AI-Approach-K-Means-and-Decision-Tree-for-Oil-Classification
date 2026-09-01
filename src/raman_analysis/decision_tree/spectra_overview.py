"""Mean-spectrum-per-oil overview plot (pure oils dataset only).

This is a data-exploration figure, not part of the classification
pipeline itself: one broken-x-axis row per oil type, showing that
type's mean Raman spectrum across all its samples. It only makes sense
for the pure-oil dataset (the chip datasets mix multiple oils per
sample matrix), which is why it is not part of the shared pipeline in
``decision_tree/pipeline.py``.

The plotted values are column-wise Z-scores, not raw intensity counts
(see ``decision_tree/config.py``'s module docstring) - the y-axis label
says so explicitly, and negative/zero values are left as-is rather than
shifted, since they are a normal, statistically meaningful part of that
normalization.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..config import LEFT_REGION_MAX, RIGHT_REGION_MIN
from ..plotting.broken_axis import add_break_marks


def plot_mean_spectra_by_oil(
    df_oils: pd.DataFrame,
    spectral_columns: list[str],
    out_path: Path,
    dpi: int = 600,
) -> None:
    """One broken-axis row per oil type, plotting its mean spectrum."""
    wavenumbers = pd.Index(spectral_columns).astype(float)
    mask_left = wavenumbers <= LEFT_REGION_MAX
    mask_right = wavenumbers >= RIGHT_REGION_MIN

    oil_types = df_oils["Oil"].unique()
    fig, axes = plt.subplots(
        len(oil_types), 2, figsize=(10, 8), sharey="row",
        gridspec_kw={"width_ratios": [4, 1]},
    )

    for row, oil in enumerate(oil_types):
        mean_spectrum = df_oils.loc[df_oils["Oil"] == oil, spectral_columns].mean(axis=0)
        y = mean_spectrum.values

        axes[row, 0].scatter(wavenumbers[mask_left], y[mask_left], s=2, label=f"Oil type: {oil}")
        axes[row, 1].scatter(wavenumbers[mask_right], y[mask_right], s=2)
        axes[row, 0].set_ylabel("Standardized Intensity (z-score)")
        axes[row, 0].legend(loc="upper right")
        axes[row, 0].grid(alpha=0.3)
        axes[row, 1].grid(alpha=0.3)
        axes[row, 1].tick_params(labelleft=False)

        add_break_marks(axes[row, 0], axes[row, 1])

    axes[-1, 0].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[-1, 1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[0, 0].set_title("Mean Raman Spectra for Each Oil Type")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
