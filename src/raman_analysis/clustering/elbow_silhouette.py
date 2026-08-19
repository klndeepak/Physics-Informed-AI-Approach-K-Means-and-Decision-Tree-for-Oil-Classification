"""Elbow (WCSS) and silhouette-score sweeps for choosing K.

The original notebooks computed each sweep twice - once to drive its own
plot with a per-k progress print, and again (identically: same data,
same seed, same range) just to stash the results under differently
named variables for a later cross-dataset comparison chart. Since
K-Means here is deterministic given a fixed ``random_state``, re-running
it a second time can only ever reproduce the same numbers, so each sweep
is computed once here and its result reused wherever the notebooks used
to recompute it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ELBOW_K_RANGE = range(2, 11)
SILHOUETTE_K_RANGE = range(2, 10)


def compute_wcss_curve(scaled_df: pd.DataFrame, random_state: int) -> list[float]:
    """Within-cluster sum of squares for each k in :data:`ELBOW_K_RANGE`."""
    wcss = []
    for k in ELBOW_K_RANGE:
        model = KMeans(n_clusters=k, random_state=random_state)
        model.fit(scaled_df)
        wcss.append(model.inertia_)
        print("Number of Clusters:", k, "\twcss:", model.inertia_)
    return wcss


def plot_elbow_curve(wcss: list[float], out_path, dpi: int = 300) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(list(ELBOW_K_RANGE), wcss, s=80, color="blue", marker="o")
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("WCSS", fontsize=12)
    plt.title("Elbow Method for Optimal k", fontsize=14)
    plt.xticks(list(ELBOW_K_RANGE))
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def compute_silhouette_curve(scaled_df: pd.DataFrame, random_state: int) -> list[float]:
    """Silhouette score for each k in :data:`SILHOUETTE_K_RANGE`."""
    scores = []
    for k in SILHOUETTE_K_RANGE:
        clusterer = KMeans(n_clusters=k, random_state=random_state)
        cluster_labels = clusterer.fit_predict(scaled_df)
        score = silhouette_score(scaled_df, cluster_labels)
        scores.append(score)
        print(f"For n_clusters = {k}, the silhouette score is {score}")
    return scores


def plot_silhouette_curve(scores: list[float], out_path, dpi: int = 300) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(list(SILHOUETTE_K_RANGE), scores, s=80, color="blue", marker="o")
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.title("Silhouette Analysis", fontsize=14)
    plt.xticks(list(SILHOUETTE_K_RANGE))
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
