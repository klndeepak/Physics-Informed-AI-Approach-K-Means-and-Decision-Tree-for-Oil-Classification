"""Regression tests for the canonical notebook t-SNE embeddings."""

import numpy as np
import pandas as pd
import pytest
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score

from raman_analysis.clustering import config, tsne_plots
from raman_analysis.clustering.scaling import scale_features
from raman_analysis.config import CLUSTERING_RANDOM_STATE
from raman_analysis.data import load_spectral_dataset, split_meta_and_spectral_columns


@pytest.mark.parametrize("dataset", [config.OILS, config.CHIPS], ids=lambda item: item.key)
@pytest.mark.parametrize("n_components", [2, 3])
def test_tsne_embedding_matches_canonical_notebook_output(dataset, n_components):
    # No shift applied - see clustering/pipeline.py: `frame`'s Z-scored
    # spectral values are left as-is, since scale_features() below
    # re-centers every column regardless of any constant offset, making a
    # shift invisible to the actual t-SNE input either way.
    frame = load_spectral_dataset(dataset.csv_path, dataset.drop_unnamed_index)
    _, spectral_columns = split_meta_and_spectral_columns(list(frame.columns))
    features = frame.drop(columns=dataset.drop_columns).astype(float)

    scaled_features = scale_features(features)
    actual = tsne_plots.run_tsne(
        scaled_features,
        n_components=n_components,
        perplexity=tsne_plots.FIXED_PERPLEXITY,
        random_state=CLUSTERING_RANDOM_STATE,
    )
    filename = dataset.filenames[f"tsne_{n_components}d_csv"]
    actual[dataset.hue_column_name] = frame[dataset.target_column].reset_index(drop=True)
    if n_components == 3:
        actual = actual[
            [dataset.hue_column_name, "Feature 1", "Feature 2", "Feature 3"]
        ]

    expected_path = config.CLUSTER_OUTPUT_DIR / filename
    if n_components == 2:
        # The 2D Barnes-Hut embedding is byte-for-byte stable across the
        # supported Windows and Linux environments, so retain the strongest
        # possible regression check for the canonical notebook output.
        expected_csv = expected_path.read_text(encoding="utf-8")
        assert actual.to_csv(index=False, lineterminator="\n") == expected_csv
        return

    # A fixed random seed does not make 3D t-SNE coordinates portable across
    # operating systems and numerical backends: tiny floating-point changes
    # during the non-convex optimization can produce a rotated, reflected, or
    # otherwise different low-dimensional layout. Exact coordinate equality
    # therefore rejected valid Linux CI output generated from the same pinned
    # inputs and parameters. Compare the scientifically relevant invariants
    # instead: sample/label identity, neighborhood preservation relative to
    # the original spectra, and class separation relative to the canonical
    # notebook embedding.
    expected = pd.read_csv(expected_path)
    label_column = dataset.hue_column_name
    feature_columns = ["Feature 1", "Feature 2", "Feature 3"]

    assert list(actual[label_column]) == list(expected[label_column])
    assert actual[feature_columns].shape == expected[feature_columns].shape
    assert np.isfinite(actual[feature_columns].to_numpy()).all()

    actual_trust = trustworthiness(
        scaled_features, actual[feature_columns], n_neighbors=5
    )
    expected_trust = trustworthiness(
        scaled_features, expected[feature_columns], n_neighbors=5
    )
    assert actual_trust >= expected_trust - 0.02

    actual_silhouette = silhouette_score(actual[feature_columns], actual[label_column])
    expected_silhouette = silhouette_score(
        expected[feature_columns], expected[label_column]
    )
    assert actual_silhouette >= expected_silhouette - 0.05
