"""Tests proving that the explicit column-wise standardization added to
the Decision Tree pipeline (see ``raman_analysis.data.standardize_train_test``)
changes only feature *units*, never a Decision Tree's predictions,
accuracy, or feature-importance ranking.

A Decision Tree's splits are threshold cuts on one feature at a time; any
per-feature affine rescaling (which z-scoring is) preserves each
feature's sample rank order, so it cannot change which side of a split
any sample falls on. These tests verify that guarantee directly on a
synthetic dataset, rather than leaving it as an unverified claim in a
comment - see decision_tree/pipeline.py and decision_tree/four_feature.py
for where the real pipeline relies on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, export_text

from raman_analysis.data import standardize_train_test

RANDOM_STATE = 0


@pytest.fixture
def toy_train_test():
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=RANDOM_STATE,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    y = pd.Series(y)
    split = int(0.7 * len(X))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def test_standardize_train_test_zeroes_training_mean_and_unit_variance(toy_train_test):
    X_train, X_test, _, _ = toy_train_test

    train_scaled, test_scaled = standardize_train_test(X_train, X_test)

    assert np.allclose(train_scaled.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(train_scaled.std(axis=0, ddof=0), 1.0, atol=1e-10)
    assert list(train_scaled.columns) == list(X_train.columns)
    assert list(test_scaled.columns) == list(X_test.columns)
    # Fit on train only: the transformed test split is not itself forced
    # to mean 0 / std 1 (it would be, coincidentally close, only if train
    # and test happened to share the same distribution exactly).
    assert train_scaled.index.equals(X_train.index)
    assert test_scaled.index.equals(X_test.index)


def test_standardize_train_test_does_not_mutate_inputs(toy_train_test):
    X_train, X_test, _, _ = toy_train_test
    X_train_copy, X_test_copy = X_train.copy(), X_test.copy()

    standardize_train_test(X_train, X_test)

    pd.testing.assert_frame_equal(X_train, X_train_copy)
    pd.testing.assert_frame_equal(X_test, X_test_copy)


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_decision_tree_predictions_are_invariant_to_standardization(
    toy_train_test, class_weight
):
    # class_weight=None matches the baseline model in
    # decision_tree/pipeline.py; "balanced" matches the pre-pruned and
    # post-pruned models actually used for every reported result - both
    # are covered since class_weight only reweights the impurity
    # criterion's class counts, never the feature values a split
    # threshold is chosen from, so it cannot interact with scaling.
    X_train, X_test, y_train, y_test = toy_train_test
    X_train_scaled, X_test_scaled = standardize_train_test(X_train, X_test)

    raw_model = DecisionTreeClassifier(
        random_state=RANDOM_STATE, class_weight=class_weight
    ).fit(X_train, y_train)
    scaled_model = DecisionTreeClassifier(
        random_state=RANDOM_STATE, class_weight=class_weight
    ).fit(X_train_scaled, y_train)

    assert list(raw_model.predict(X_test)) == list(scaled_model.predict(X_test_scaled))
    np.testing.assert_array_equal(
        raw_model.predict(X_train), scaled_model.predict(X_train_scaled)
    )
    np.testing.assert_allclose(
        raw_model.feature_importances_, scaled_model.feature_importances_
    )
    assert raw_model.get_n_leaves() == scaled_model.get_n_leaves()
    assert raw_model.get_depth() == scaled_model.get_depth()


def test_decision_tree_structure_is_identical_up_to_threshold_units(toy_train_test):
    # Same split points in rank order => same tree shape and the same
    # feature chosen at every node; only the printed numeric threshold
    # differs (raw value vs. z-score), which is expected and is exactly
    # the units-only change the docstrings describe.
    X_train, X_test, y_train, y_test = toy_train_test
    X_train_scaled, _ = standardize_train_test(X_train, X_test)

    raw_model = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
    scaled_model = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(
        X_train_scaled, y_train
    )

    feature_names = list(X_train.columns)
    raw_report = export_text(raw_model, feature_names=feature_names)
    scaled_report = export_text(scaled_model, feature_names=feature_names)

    def _strip_numeric_thresholds(report: str) -> list[str]:
        lines = []
        for line in report.splitlines():
            for operator in ("<=", ">"):
                if operator in line:
                    line = line.split(operator)[0] + operator
                    break
            lines.append(line)
        return lines

    assert _strip_numeric_thresholds(raw_report) == _strip_numeric_thresholds(scaled_report)
