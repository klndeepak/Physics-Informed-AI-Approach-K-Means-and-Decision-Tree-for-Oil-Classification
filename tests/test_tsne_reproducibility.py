"""Regression tests for the canonical notebook t-SNE coordinates."""

import pytest

from raman_analysis.clustering import config, tsne_plots
from raman_analysis.clustering.scaling import scale_features
from raman_analysis.config import CLUSTERING_RANDOM_STATE
from raman_analysis.data import load_spectral_dataset, split_meta_and_spectral_columns


@pytest.mark.parametrize("dataset", [config.OILS, config.CHIPS], ids=lambda item: item.key)
@pytest.mark.parametrize("n_components", [2, 3])
def test_tsne_coordinates_match_canonical_notebook_output(dataset, n_components):
    # No shift applied - see clustering/pipeline.py: `frame`'s Z-scored
    # spectral values are left as-is, since scale_features() below
    # re-centers every column regardless of any constant offset, making a
    # shift invisible to the actual t-SNE input either way.
    frame = load_spectral_dataset(dataset.csv_path, dataset.drop_unnamed_index)
    _, spectral_columns = split_meta_and_spectral_columns(list(frame.columns))
    features = frame.drop(columns=dataset.drop_columns).astype(float)

    actual = tsne_plots.run_tsne(
        scale_features(features),
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

    expected_csv = (config.CLUSTER_OUTPUT_DIR / filename).read_text(encoding="utf-8")
    assert actual.to_csv(index=False, lineterminator="\n") == expected_csv
