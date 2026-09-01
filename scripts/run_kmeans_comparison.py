#!/usr/bin/env python3
"""Oils-vs-chips elbow and silhouette comparison charts.

The original notebooks built these two figures (``K-Clusters-Combined.jpg``
and ``silhouette_combined.jpg``) by hand-copying the WCSS/silhouette
numbers computed in one notebook run into the other. Since K-Means here
is deterministic given a fixed random seed, this script instead
recomputes both sweeps directly (scaling + elbow + silhouette only - it
skips the expensive t-SNE steps, since neither comparison chart needs
them) and always plots live numbers.

Usage::

    python scripts/run_kmeans_comparison.py
"""

from raman_analysis.clustering import compare, config, elbow_silhouette
from raman_analysis.clustering.scaling import scale_features
from raman_analysis.config import CLUSTERING_RANDOM_STATE
from raman_analysis.data import (
    load_spectral_dataset,
    report_spectrum_minimum,
    split_meta_and_spectral_columns,
)
from raman_analysis.paths import ensure_dir


def _scaled_features(dataset):
    df = load_spectral_dataset(dataset.csv_path, dataset.drop_unnamed_index)
    _, spectral_columns = split_meta_and_spectral_columns(list(df.columns))

    # Match clustering.pipeline: negative values are meaningful Z-scores,
    # and scale_features() re-centers every column anyway, so a constant
    # nonnegative shift would have no effect on either comparison curve.
    report_spectrum_minimum(df[spectral_columns])

    X = df.drop(columns=dataset.drop_columns).astype(float)
    return scale_features(X)


def main() -> None:
    output_dir = ensure_dir(config.CLUSTER_OUTPUT_DIR)
    random_state = CLUSTERING_RANDOM_STATE

    oils_scaled = _scaled_features(config.OILS)
    chips_scaled = _scaled_features(config.CHIPS)

    oils_wcss = elbow_silhouette.compute_wcss_curve(oils_scaled, random_state)
    chips_wcss = elbow_silhouette.compute_wcss_curve(chips_scaled, random_state)
    compare.plot_combined_wcss(oils_wcss, chips_wcss, output_dir / "K-Clusters-Combined.jpg")

    oils_silhouette = elbow_silhouette.compute_silhouette_curve(oils_scaled, random_state)
    chips_silhouette = elbow_silhouette.compute_silhouette_curve(chips_scaled, random_state)
    compare.plot_combined_silhouette(
        oils_silhouette, chips_silhouette, output_dir / "silhouette_combined.jpg"
    )


if __name__ == "__main__":
    main()
