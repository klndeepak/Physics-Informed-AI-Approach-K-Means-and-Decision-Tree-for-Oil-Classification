"""Tests for raman_analysis.decision_tree.pruning.

Uses a small synthetic 3-class dataset (not the real spectra) purely to
exercise the grid-search and cost-complexity-pruning plumbing quickly.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from raman_analysis.decision_tree import pruning

RANDOM_STATE = 0


@pytest.fixture
def toy_dataset():
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=RANDOM_STATE,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), pd.Series(y)


def test_grid_search_pre_pruned_returns_a_fitted_tree(toy_dataset):
    X, y = toy_dataset

    model, cv_results = pruning.grid_search_pre_pruned(X, y, RANDOM_STATE)

    assert hasattr(model, "predict")
    assert len(model.predict(X)) == len(X)
    assert list(cv_results.columns) == [
        "params", "mean_test_score", "std_test_score", "rank_test_score",
    ]
    assert cv_results["rank_test_score"].iloc[0] == cv_results["rank_test_score"].min()


def test_cost_complexity_pruning_path_is_nonnegative_and_increasing(toy_dataset):
    X, y = toy_dataset

    ccp_alphas, impurities = pruning.cost_complexity_pruning_path(X, y, RANDOM_STATE)

    assert (ccp_alphas >= 0).all()
    assert np.all(np.diff(ccp_alphas) >= 0)
    assert len(ccp_alphas) == len(impurities)


def test_fit_pruned_tree_sequence_collapses_to_a_single_leaf(toy_dataset):
    X, y = toy_dataset
    ccp_alphas, _ = pruning.cost_complexity_pruning_path(X, y, RANDOM_STATE)

    trees = pruning.fit_pruned_tree_sequence(X, y, ccp_alphas, RANDOM_STATE)

    assert len(trees) == len(ccp_alphas)
    assert trees[-1].get_n_leaves() == 1


def test_weighted_recall_and_accuracy_curves_are_bounded(toy_dataset):
    X, y = toy_dataset
    ccp_alphas, _ = pruning.cost_complexity_pruning_path(X, y, RANDOM_STATE)
    trees = pruning.fit_pruned_tree_sequence(X, y, ccp_alphas, RANDOM_STATE)

    recalls = pruning.weighted_recall_curve(trees, X, y)
    accuracies = pruning.accuracy_curve(trees, X, y)

    assert all(0.0 <= value <= 1.0 for value in recalls)
    assert all(0.0 <= value <= 1.0 for value in accuracies)
