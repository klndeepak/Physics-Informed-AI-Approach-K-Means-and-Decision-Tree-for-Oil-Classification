#!/usr/bin/env python3
"""K-Means / t-SNE clustering of the raw fried-chip Raman spectra.

Reproduces ``K-Means Clusters_Chips.ipynb``: the same elbow/silhouette/
t-SNE/K-Means workflow as the pure oils, run on the uncorrected chip
spectra, to see how much of the oils' cluster structure survives once
the oil signal is mixed into the fried-food matrix. Figures are written
under ``Images/K-Means Clusters/``.

Usage::

    python scripts/run_kmeans_chips.py
"""

from raman_analysis.clustering import config, pipeline


def main() -> None:
    pipeline.run(config.CHIPS)


if __name__ == "__main__":
    main()
