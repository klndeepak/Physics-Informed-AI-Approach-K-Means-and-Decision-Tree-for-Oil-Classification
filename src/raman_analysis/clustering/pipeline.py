"""End-to-end clustering pipeline: scale -> t-SNE -> elbow -> silhouette
-> K-Means -> cluster profile, for one dataset.

Runs identically for both the pure oils and the chips dataset; what
differs (palette, legend text quirks, perplexity sweep, output
filenames) comes from a single :class:`ClusteringDatasetConfig`. See
that module's docstring for the ``hue_column_name`` naming quirk this
pipeline reproduces from the original notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.manifold import TSNE

from . import cluster_profile, elbow_silhouette, kmeans_clusters, tsne_plots
from .config import CLUSTER_OUTPUT_DIR, ClusteringDatasetConfig, N_OIL_CLASSES
from .scaling import scale_features
from ..config import CLUSTERING_RANDOM_STATE
from ..data import (
    load_spectral_dataset,
    report_spectrum_minimum,
    shift_to_nonnegative,
    split_meta_and_spectral_columns,
)
from ..paths import ensure_dir


@dataclass
class ClusteringResult:
    """The two sweeps other scripts need to build cross-dataset charts."""

    wcss: list[float]
    silhouette_scores: list[float]


def run(dataset: ClusteringDatasetConfig) -> ClusteringResult:
    random_state = CLUSTERING_RANDOM_STATE
    output_dir = ensure_dir(CLUSTER_OUTPUT_DIR)
    files = dataset.filenames

    df = load_spectral_dataset(dataset.csv_path, dataset.drop_unnamed_index)
    _, spectral_columns = split_meta_and_spectral_columns(list(df.columns))

    min_value, _, _ = report_spectrum_minimum(df[spectral_columns])
    df = shift_to_nonnegative(df, spectral_columns, min_value)

    X = df.drop(columns=dataset.drop_columns).astype(float)
    y = df[dataset.target_column]
    scaled = scale_features(X)

    tsne_2d = tsne_plots.run_tsne(
        scaled, n_components=2, perplexity=tsne_plots.DEFAULT_PERPLEXITY, random_state=random_state
    )
    tsne_plots.plot_tsne_default(
        tsne_2d, y, dataset.hue_column_name, dataset.palette, dataset.tsne_default_legend_loc,
        output_dir / files["tsne_default"],
    )
    print(TSNE().get_params()["perplexity"])

    tsne_plots.plot_perplexity_grid(
        scaled, y, dataset.hue_column_name, dataset.hue_order, dataset.palette,
        dataset.perplexities, random_state, output_dir / files["tsne_perplexity_grid"],
    )

    tsne_2d = tsne_plots.run_tsne(
        scaled, n_components=2, perplexity=tsne_plots.FIXED_PERPLEXITY, random_state=random_state
    )
    tsne_2d[dataset.hue_column_name] = y.reset_index(drop=True)
    tsne_plots.save_and_verify_csv(tsne_2d, output_dir / files["tsne_2d_csv"])
    tsne_plots.print_feature_ranges(tsne_2d, dataset.hue_column_name, ["Feature 1", "Feature 2"])

    tsne_3d = tsne_plots.run_tsne(
        scaled, n_components=3, perplexity=tsne_plots.FIXED_PERPLEXITY, random_state=random_state
    )
    tsne_3d[dataset.hue_column_name] = y.reset_index(drop=True)
    tsne_3d = tsne_3d[[dataset.hue_column_name, "Feature 1", "Feature 2", "Feature 3"]]
    tsne_plots.save_and_verify_csv(tsne_3d, output_dir / files["tsne_3d_csv"])
    tsne_plots.print_feature_ranges(
        tsne_3d, dataset.hue_column_name, ["Feature 1", "Feature 2", "Feature 3"]
    )

    tsne_plots.plot_pairwise_projections(
        tsne_3d, dataset.hue_column_name, dataset.palette, output_dir / files["tsne_3d_pairwise"]
    )
    tsne_plots.plot_tsne_3d_plotly(
        tsne_3d, y, dataset.palette_flat, output_dir / files["tsne_3d_plotly"]
    )

    wcss = elbow_silhouette.compute_wcss_curve(scaled, random_state)
    elbow_silhouette.plot_elbow_curve(wcss, output_dir / files["elbow_plot"])
    print("K-Means:", list(elbow_silhouette.ELBOW_K_RANGE))
    print(f"WCSS_{dataset.key.title()}:", wcss)

    silhouette_scores = elbow_silhouette.compute_silhouette_curve(scaled, random_state)
    elbow_silhouette.plot_silhouette_curve(
        silhouette_scores, output_dir / files["silhouette_plot"]
    )
    print("n_clusters:", list(elbow_silhouette.SILHOUETTE_K_RANGE))
    print(f"sil_{dataset.key}:", silhouette_scores)

    kmeans = kmeans_clusters.fit_final_kmeans(scaled, N_OIL_CLASSES, random_state)
    mapping = kmeans_clusters.map_clusters_to_oil_labels(y, kmeans.labels_)

    tsne_2d["K_means_segments"] = kmeans.labels_
    tsne_2d["Cluster_Label"] = pd.Series(kmeans.labels_).map(mapping)
    print(tsne_2d["Cluster_Label"].value_counts())

    kmeans_clusters.print_cluster_vs_label_report(y, kmeans.labels_)
    tsne_2d["True_Label"] = y.values
    print(tsne_2d.columns)

    kmeans_clusters.plot_true_vs_cluster(
        tsne_2d, dataset.hue_column_name, dataset.palette,
        output_dir / files["kmeans_tsne_clusters"],
    )

    cluster_profile_df = kmeans_clusters.build_cluster_profile(X, kmeans.labels_, mapping)
    cluster_profile.plot_cluster_profiles(
        cluster_profile_df, dataset.palette_flat, output_dir / files["cluster_profiles"]
    )

    if dataset.has_average_vs_random_plot:
        df_with_clusters = X.copy()
        df_with_clusters["K_means_segments"] = kmeans.labels_
        oil_to_cluster = {oil: cluster_id for cluster_id, oil in mapping.items()}
        cluster_profile.plot_cluster_average_vs_random(
            df_with_clusters, cluster_profile_df, oil_to_cluster, dataset.palette_flat,
            output_dir / files["cluster_average_vs_random"],
        )

    return ClusteringResult(wcss=wcss, silhouette_scores=silhouette_scores)
