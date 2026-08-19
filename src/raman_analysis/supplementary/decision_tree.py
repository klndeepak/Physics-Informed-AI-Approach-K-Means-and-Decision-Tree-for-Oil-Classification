"""Decision-tree tables used by the additional-results runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ..config import DECISION_TREE_RANDOM_STATE
from ..data import load_spectral_dataset
from ..decision_tree import pruning
from ..decision_tree.pipeline import TEST_SIZE
from ..metrics import classification_performance

ROBUSTNESS_SEEDS = tuple(range(10))


def _load_xy(dataset):
    frame = load_spectral_dataset(dataset.csv_path, dataset.drop_unnamed_index)
    return frame.drop(columns=dataset.drop_columns).astype(float), frame[dataset.target_column]


def _split(X, y, seed):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)


def _fit_canonical_models(X_train, y_train):
    seed = DECISION_TREE_RANDOM_STATE
    baseline = DecisionTreeClassifier(random_state=seed).fit(X_train, y_train)
    pre_pruned, _ = pruning.grid_search_pre_pruned(X_train, y_train, seed)
    alphas, impurities = pruning.cost_complexity_pruning_path(X_train, y_train, seed)
    post_pruned, _ = pruning.grid_search_post_pruned(X_train, y_train, alphas, seed)
    return {
        "baseline": baseline,
        "pre_pruned": pre_pruned,
        "post_pruned": post_pruned,
    }, alphas, impurities


def _metric_row(dataset, model_name, split_name, model, X, y):
    values = classification_performance(model, X, y).iloc[0]
    return {
        "dataset": dataset.key,
        "model": model_name,
        "split": split_name,
        "accuracy": values["Accuracy"],
        "recall_weighted": values["Recall"],
        "precision_weighted": values["Precision"],
        "f1_weighted": values["F1"],
        "tree_depth": model.get_depth(),
        "leaf_count": model.get_n_leaves(),
        "selected_feature_count": int(np.count_nonzero(model.feature_importances_)),
    }


def _confusion_rows(dataset, model_name, split_name, model, X, y):
    labels = sorted(y.unique())
    matrix = confusion_matrix(y, model.predict(X), labels=labels)
    return [
        {
            "dataset": dataset.key,
            "model": model_name,
            "split": split_name,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": int(matrix[i, j]),
        }
        for i, true_label in enumerate(labels)
        for j, predicted_label in enumerate(labels)
    ]


def _model_from_template(name, template, seed):
    if name == "baseline":
        return DecisionTreeClassifier(random_state=seed)
    if name == "pre_pruned":
        return DecisionTreeClassifier(
            class_weight="balanced",
            random_state=seed,
            max_depth=template.max_depth,
            max_leaf_nodes=template.max_leaf_nodes,
            min_samples_split=template.min_samples_split,
        )
    return DecisionTreeClassifier(
        class_weight="balanced", random_state=seed, ccp_alpha=template.ccp_alpha
    )


def collect(datasets):
    metrics_rows = []
    confusion_rows = []
    pruning_rows = []
    robustness_rows = []
    feature_rows = []

    for dataset in datasets:
        print(f"Collecting Decision Tree results: {dataset.key}")
        X, y = _load_xy(dataset)
        X_train, X_test, y_train, y_test = _split(X, y, DECISION_TREE_RANDOM_STATE)
        models, alphas, impurities = _fit_canonical_models(X_train, y_train)

        for model_name, model in models.items():
            for split_name, split_X, split_y in (
                ("train", X_train, y_train),
                ("test", X_test, y_test),
            ):
                metrics_rows.append(
                    _metric_row(dataset, model_name, split_name, model, split_X, split_y)
                )
                confusion_rows.extend(
                    _confusion_rows(dataset, model_name, split_name, model, split_X, split_y)
                )

        trees = pruning.fit_pruned_tree_sequence(
            X_train, y_train, alphas, DECISION_TREE_RANDOM_STATE
        )
        for alpha, impurity, tree in zip(alphas, impurities, trees):
            pruning_rows.append(
                {
                    "dataset": dataset.key,
                    "ccp_alpha": alpha,
                    "impurity": impurity,
                    "tree_depth": tree.get_depth(),
                    "leaf_count": tree.get_n_leaves(),
                    "node_count": tree.tree_.node_count,
                    "train_accuracy": tree.score(X_train, y_train),
                    "test_accuracy": tree.score(X_test, y_test),
                }
            )

        for seed in ROBUSTNESS_SEEDS:
            seed_X_train, seed_X_test, seed_y_train, seed_y_test = _split(X, y, seed)
            for model_name, template in models.items():
                model = _model_from_template(model_name, template, seed)
                model.fit(seed_X_train, seed_y_train)
                row = _metric_row(
                    dataset, model_name, "test", model, seed_X_test, seed_y_test
                )
                row["seed"] = seed
                robustness_rows.append(row)
                for feature, importance in zip(X.columns, model.feature_importances_):
                    feature_rows.append(
                        {
                            "dataset": dataset.key,
                            "model": model_name,
                            "seed": seed,
                            "wavenumber": feature,
                            "importance": importance,
                            "selected": int(importance > 0),
                        }
                    )

    feature_runs = pd.DataFrame(feature_rows)
    feature_summary = (
        feature_runs.groupby(["dataset", "model", "wavenumber"], as_index=False)
        .agg(
            selection_count=("selected", "sum"),
            selection_rate=("selected", "mean"),
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
        )
        .sort_values(
            ["dataset", "model", "selection_rate", "mean_importance"],
            ascending=[True, True, False, False],
        )
    )
    return {
        "model_metrics_summary": pd.DataFrame(metrics_rows),
        "confusion_matrices": pd.DataFrame(confusion_rows),
        "pruning_results": pd.DataFrame(pruning_rows),
        "model_robustness": pd.DataFrame(robustness_rows),
        "feature_stability": feature_summary,
    }
