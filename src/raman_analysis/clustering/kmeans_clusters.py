"""Final K-Means model (K = number of oil classes) and its evaluation
against the true oil labels.

K-Means is unsupervised - it never sees ``y`` - so afterward each numeric
cluster ID is mapped to the oil label it overlaps with most (its
majority vote via a crosstab), purely so results can be *read* against
known ground truth. The crosstab itself is the real evidence of how well
clusters and true labels line up; the mapping is only a labeling
convenience for the plots and profiles built on top of it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


def fit_final_kmeans(scaled_df: pd.DataFrame, n_clusters: int, random_state: int) -> KMeans:
    """Fit K-Means on a string-column copy of ``scaled_df`` (sklearn wants
    homogeneous, hashable column labels; the wavenumber columns are floats).
    """
    fit_df = scaled_df.copy()
    fit_df.columns = fit_df.columns.astype(str)
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(fit_df)
    return model


def map_clusters_to_oil_labels(y: pd.Series, cluster_labels) -> dict[int, str]:
    """Majority-vote oil label for each numeric cluster ID."""
    return pd.crosstab(y, cluster_labels).idxmax().to_dict()


def print_cluster_vs_label_report(y: pd.Series, cluster_labels) -> None:
    cross_tab = pd.crosstab(y, cluster_labels, margins=True)
    print("\n=== Cluster vs True Label Comparison ===")
    print(cross_tab)

    dominant_cluster = cross_tab.drop("All", axis=1).idxmax(axis=1)
    print("\n=== Dominant Cluster for Each Oil Type ===")
    print(dominant_cluster)


def plot_true_vs_cluster(
    tsne_2d: pd.DataFrame,
    hue_column_name: str,
    palette: dict[str, str],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Side-by-side t-SNE scatter: true oil label vs. assigned cluster label."""
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(
        data=tsne_2d, x="Feature 1", y="Feature 2", hue=hue_column_name, palette=palette
    )
    plt.title("t-SNE (True Labels)")

    plt.subplot(1, 2, 2)
    sns.scatterplot(
        data=tsne_2d, x="Feature 1", y="Feature 2", hue="Cluster_Label", palette=palette
    )
    plt.title("K-Means Clusters Visualized using t-SNE Projection onto 2D")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def build_cluster_profile(
    df_unscaled: pd.DataFrame, cluster_labels, mapping: dict[int, str]
) -> pd.DataFrame:
    """Mean spectrum per cluster, with a sample count and oil-label index."""
    df_unscaled = df_unscaled.copy()
    df_unscaled["K_means_segments"] = cluster_labels

    profile = df_unscaled.groupby("K_means_segments").mean(numeric_only=True)
    profile["count_in_each_segment"] = df_unscaled.groupby("K_means_segments").size().values
    profile.index = profile.index.map(mapping)
    return profile
