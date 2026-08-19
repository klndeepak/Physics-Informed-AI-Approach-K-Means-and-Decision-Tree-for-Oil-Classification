"""Broken x-axis helper for spectral plots.

Three figures in this project (the per-oil mean-spectrum overview, and
two K-Means cluster-profile plots) split each row's wavenumber axis into
a "fingerprint region" panel and a "CH-stretch region" panel, with the
uninformative gap between :data:`config.LEFT_REGION_MAX` and
:data:`config.RIGHT_REGION_MIN` visually broken out. This module holds
the one piece of drawing logic identical across all three: the small
diagonal "break" marks at the seam between the two panels.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def add_break_marks(ax_left: plt.Axes, ax_right: plt.Axes, d: float = 0.015) -> None:
    """Draw diagonal break marks where ``ax_left`` and ``ax_right`` meet.

    Also hides the touching spines (right edge of the left panel, left
    edge of the right panel) so the two panels read as one broken axis
    rather than two separate plots.
    """
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)

    kwargs = dict(transform=ax_left.transAxes, color="k", clip_on=False)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-d, +d), (-d, +d), **kwargs)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)
