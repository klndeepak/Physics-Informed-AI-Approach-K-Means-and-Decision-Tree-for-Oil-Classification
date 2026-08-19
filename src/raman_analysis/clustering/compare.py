"""Cross-dataset comparison charts: pure oils vs. fried chips.

These two figures overlay the oils and chips elbow/silhouette sweeps to
show that intrinsic cluster structure is stronger for the pure oils than
for the chip matrix. They need both datasets' sweep results, so they run
after both single-dataset pipelines (see ``scripts/run_kmeans_oils.py``
and ``scripts/run_kmeans_chips.py``) rather than as part of either one.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .elbow_silhouette import ELBOW_K_RANGE, SILHOUETTE_K_RANGE


def plot_combined_wcss(
    oils_wcss: list[float], chips_wcss: list[float], out_path, dpi: int = 300
) -> None:
    k_values = list(ELBOW_K_RANGE)
    plt.figure(figsize=(10, 6))
    plt.scatter(k_values, oils_wcss, color="black", label="Oils")
    plt.scatter(k_values, chips_wcss, color="blue", label="Chips")
    plt.xlabel("K-Means")
    plt.ylabel("WCSS")
    plt.title("WCSS vs Number of Clusters")
    plt.xticks(k_values)
    plt.legend()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_combined_silhouette(
    oils_scores: list[float], chips_scores: list[float], out_path, dpi: int = 300
) -> None:
    k_values = list(SILHOUETTE_K_RANGE)
    plt.figure(figsize=(10, 6))
    plt.scatter(k_values, oils_scores, color="black", label="Oils")
    plt.scatter(k_values, chips_scores, color="blue", label="Chips")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs Number of Clusters")
    plt.xticks(k_values)
    plt.legend()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
