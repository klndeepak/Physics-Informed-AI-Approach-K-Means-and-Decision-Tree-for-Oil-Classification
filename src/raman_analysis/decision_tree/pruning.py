"""Pre-pruning and cost-complexity post-pruning for Decision Trees.

Both pruning strategies are selected the same way in every notebook:
5-fold, stratified, cross-validated weighted recall on the training set
only. The test set is never touched until the winning model of each
stage is scored once, at the end, in ``decision_tree/pipeline.py``.

- **Pre-pruning** restricts tree growth *during* fitting by grid-searching
  ``max_depth``, ``max_leaf_nodes``, and ``min_samples_split``.
- **Post-pruning** grows one full tree, reads off its cost-complexity
  pruning path (a finite sequence of ``ccp_alpha`` values), and grid
  searches over that sequence.

``class_weight="balanced"`` is used everywhere here so a rarer class
counts for more in the split criterion, guarding against any imbalance
the train/test split introduces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

CV_FOLDS = 5
CV_SCORING = "recall_weighted"

PRE_PRUNING_GRID = {
    "max_depth": list(np.arange(2, 7, 2)),
    "max_leaf_nodes": [50, 75, 150, 250],
    "min_samples_split": [10, 30, 50, 70],
}

_CV_RESULT_COLUMNS = ["params", "mean_test_score", "std_test_score", "rank_test_score"]


def _stratified_cv(random_state: int) -> StratifiedKFold:
    # Stratifies by class label, evaluating at the spectrum level - the
    # same scope documented in decision_tree/pipeline.py's "SCOPE" note
    # applies to these CV folds too, consistently with the rest of this
    # study's spectrum-level evaluation.
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)


def grid_search_pre_pruned(
    X_train, y_train, random_state: int
) -> tuple[DecisionTreeClassifier, pd.DataFrame]:
    """Grid search max_depth / max_leaf_nodes / min_samples_split.

    Returns the best-scoring estimator (refit on the full training set)
    and a CV results table sorted by rank.
    """
    search = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
        param_grid=PRE_PRUNING_GRID,
        scoring=CV_SCORING,
        cv=_stratified_cv(random_state),
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    cv_results = (
        pd.DataFrame(search.cv_results_)[_CV_RESULT_COLUMNS]
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )
    return search.best_estimator_, cv_results


def cost_complexity_pruning_path(
    X_train, y_train, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Effective alphas and leaf impurities for a fully grown tree.

    ``ccp_alphas`` is a finite, increasing sequence: alpha=0 is the full
    tree, and each subsequent value is where the next weakest-link node
    collapses. abs() guards against a stray floating-point sign flip on
    a value that should be exactly zero.
    """
    full_tree = DecisionTreeClassifier(random_state=random_state, class_weight="balanced")
    path = full_tree.cost_complexity_pruning_path(X_train, y_train)
    return np.abs(path.ccp_alphas), path.impurities


def fit_pruned_tree_sequence(
    X_train, y_train, ccp_alphas: np.ndarray, random_state: int
) -> list[DecisionTreeClassifier]:
    """Fit one tree per alpha in ``ccp_alphas``, heaviest pruning last."""
    trees = []
    for ccp_alpha in ccp_alphas:
        tree = DecisionTreeClassifier(
            random_state=random_state, ccp_alpha=ccp_alpha, class_weight="balanced"
        )
        tree.fit(X_train, y_train)
        trees.append(tree)
    return trees


def grid_search_post_pruned(
    X_train, y_train, ccp_alphas: np.ndarray, random_state: int
) -> tuple[DecisionTreeClassifier, pd.DataFrame]:
    """Grid search ``ccp_alpha`` over the cost-complexity pruning path.

    Returns the best-scoring pruned tree and a CV results table sorted
    by alpha (ascending, i.e. least to most pruned).
    """
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
        param_grid={"ccp_alpha": list(ccp_alphas)},
        scoring=CV_SCORING,
        cv=_stratified_cv(random_state),
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    cv_table = pd.DataFrame(
        {
            "ccp_alpha": list(ccp_alphas),
            "mean_cv_recall": search.cv_results_["mean_test_score"],
            "std_cv_recall": search.cv_results_["std_test_score"],
        }
    ).sort_values("ccp_alpha").reset_index(drop=True)
    return search.best_estimator_, cv_table


def weighted_recall_curve(trees: list[DecisionTreeClassifier], X, y) -> list[float]:
    """Weighted recall of each tree in ``trees`` against (X, y)."""
    return [recall_score(y, tree.predict(X), average="weighted") for tree in trees]


def accuracy_curve(trees: list[DecisionTreeClassifier], X, y) -> list[float]:
    """Plain accuracy (``estimator.score``) of each tree against (X, y)."""
    return [tree.score(X, y) for tree in trees]
