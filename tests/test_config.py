"""Sanity checks on the dataset configuration registries.

These guard against the kind of copy-paste mistake that is easy to make
in a config file with many near-identical entries: a filename typo'd
into the wrong dataset's block, or two datasets accidentally sharing an
output path and silently overwriting each other's figures.
"""

from raman_analysis.clustering import config as clustering_config
from raman_analysis.decision_tree import config as decision_tree_config


def test_decision_tree_datasets_have_disjoint_output_directories():
    output_dirs = [ds.output_dir for ds in decision_tree_config.DATASETS.values()]
    assert len(output_dirs) == len(set(output_dirs))


def test_decision_tree_datasets_reference_an_existing_csv():
    for dataset in decision_tree_config.DATASETS.values():
        assert dataset.csv_path.exists(), f"{dataset.key}: {dataset.csv_path} is missing"


def test_decision_tree_filenames_are_unique_within_each_dataset():
    for dataset in decision_tree_config.DATASETS.values():
        filenames = list(dataset.filenames.values())
        assert len(filenames) == len(set(filenames)), dataset.key


def test_clustering_datasets_reference_an_existing_csv():
    for dataset in clustering_config.DATASETS.values():
        assert dataset.csv_path.exists(), f"{dataset.key}: {dataset.csv_path} is missing"


def test_clustering_datasets_share_one_output_directory_without_filename_collisions():
    # Oils and Chips both write into "Images/K-Means Clusters/" - unlike
    # the Decision Tree datasets, so their filenames must be
    # cross-dataset unique, not just unique within each dataset.
    all_filenames = [
        filename
        for dataset in clustering_config.DATASETS.values()
        for filename in dataset.filenames.values()
    ]
    assert len(all_filenames) == len(set(all_filenames))


def test_clustering_hue_order_matches_palette_keys():
    for dataset in clustering_config.DATASETS.values():
        assert set(dataset.hue_order) == set(dataset.palette.keys())
        assert set(dataset.hue_order) == set(dataset.palette_flat.keys())
