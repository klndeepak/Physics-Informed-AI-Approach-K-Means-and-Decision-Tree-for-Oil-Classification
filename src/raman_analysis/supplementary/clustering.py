"""Clustering score, contingency, and stability tables."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from ..clustering import elbow_silhouette
from ..clustering.config import N_OIL_CLASSES
from ..clustering.scaling import scale_features
from ..config import CLUSTERING_RANDOM_STATE
from ..data import load_spectral_dataset, shift_to_nonnegative, split_meta_and_spectral_columns

STABILITY_SEEDS = tuple(range(10))


def _load_scaled(dataset):
    frame = load_spectral_dataset(dataset.csv_path, dataset.drop_unnamed_index)
    _, spectral_columns = split_meta_and_spectral_columns(list(frame.columns))
    minimum = frame[spectral_columns].min().min()
    frame = shift_to_nonnegative(frame, spectral_columns, minimum)
    X = frame.drop(columns=dataset.drop_columns).astype(float)
    return scale_features(X), frame[dataset.target_column]


def collect(datasets):
    score_rows = []
    contingency_rows = []
    stability_rows = []

    for dataset in datasets:
        print(f"Collecting clustering results: {dataset.key}")
        scaled, y = _load_scaled(dataset)
        wcss = elbow_silhouette.compute_wcss_curve(scaled, CLUSTERING_RANDOM_STATE)
        silhouettes = elbow_silhouette.compute_silhouette_curve(
            scaled, CLUSTERING_RANDOM_STATE
        )
        silhouette_by_k = dict(zip(elbow_silhouette.SILHOUETTE_K_RANGE, silhouettes))
        for k, inertia in zip(elbow_silhouette.ELBOW_K_RANGE, wcss):
            score_rows.append(
                {
                    "dataset": dataset.key,
                    "k": k,
                    "wcss": inertia,
                    "silhouette_score": silhouette_by_k.get(k),
                }
            )

        reference = KMeans(
            n_clusters=N_OIL_CLASSES, random_state=CLUSTERING_RANDOM_STATE
        ).fit(scaled)
        table = pd.crosstab(y, reference.labels_)
        ari = adjusted_rand_score(y, reference.labels_)
        nmi = normalized_mutual_info_score(y, reference.labels_)
        for true_label in table.index:
            for cluster in table.columns:
                contingency_rows.append(
                    {
                        "dataset": dataset.key,
                        "true_label": true_label,
                        "cluster": int(cluster),
                        "count": int(table.loc[true_label, cluster]),
                        "adjusted_rand_index": ari,
                        "normalized_mutual_info": nmi,
                    }
                )

        for seed in STABILITY_SEEDS:
            model = KMeans(n_clusters=N_OIL_CLASSES, random_state=seed).fit(scaled)
            stability_rows.append(
                {
                    "dataset": dataset.key,
                    "seed": seed,
                    "inertia": model.inertia_,
                    "silhouette_score": silhouette_score(scaled, model.labels_),
                    "ari_vs_reference": adjusted_rand_score(
                        reference.labels_, model.labels_
                    ),
                    "nmi_vs_reference": normalized_mutual_info_score(
                        reference.labels_, model.labels_
                    ),
                }
            )

    return {
        "clustering_scores": pd.DataFrame(score_rows),
        "cluster_contingency_tables": pd.DataFrame(contingency_rows),
        "kmeans_stability": pd.DataFrame(stability_rows),
    }
