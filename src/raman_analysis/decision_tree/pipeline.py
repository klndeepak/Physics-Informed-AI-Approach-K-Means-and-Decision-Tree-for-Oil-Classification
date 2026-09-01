"""End-to-end Decision Tree pipeline: baseline -> pre-pruning -> post-pruning.

This runs identically for all four spectral datasets (see the module
docstring in ``config.py``); everything that differs between them - file
paths, dropped columns, the target column's name, output filenames - is
supplied by a single :class:`DecisionTreeDatasetConfig`.

Class tick-mark labels and tree ``class_names`` intentionally use the
fixed, plain-code list ``['GNO', 'PO', 'SO', 'SOYO', 'VO']`` for every
dataset, exactly as the original notebooks did - even for the chip
datasets, whose actual class values are suffixed "GNO C" etc. Both lists
share the same alphabetical order, so this only affects a cosmetic
suffix on plot tick labels, never which class a prediction is scored
against; it is kept as-is here to reproduce the notebooks' figures
exactly.

Every stage below (baseline, pre-pruned, post-pruned) reports accuracy,
precision, recall, F1, and a confusion matrix for BOTH the training set
and the held-out test set - never cross-validation results alone. The
5-fold ``StratifiedKFold`` cross-validation in ``pruning.py`` is used
only to select hyperparameters (``max_depth``/``max_leaf_nodes``/
``min_samples_split`` for pre-pruning, ``ccp_alpha`` for post-pruning) on
the training split; the test split is never touched until the winning
model of each stage is scored once, at the end, right here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from . import pruning, reporting
from .config import DecisionTreeDatasetConfig
from .four_feature import build_important_features_dataframe, run_four_feature_check
from .spectra_overview import plot_mean_spectra_by_oil
from ..config import DECISION_TREE_RANDOM_STATE, OIL_CODES_SORTED
from ..data import (
    load_spectral_dataset,
    report_spectrum_minimum,
    split_meta_and_spectral_columns,
    standardize_train_test,
)
from ..metrics import classification_performance, plot_confusion_matrix
from ..paths import ensure_dir

TEST_SIZE = 0.30


def run(dataset: DecisionTreeDatasetConfig) -> None:
    """Run the full baseline/pre-pruning/post-pruning workflow."""
    random_state = DECISION_TREE_RANDOM_STATE
    output_dir = ensure_dir(dataset.output_dir)
    files = dataset.filenames

    df = load_spectral_dataset(dataset.csv_path, dataset.drop_unnamed_index)
    _, spectral_columns = split_meta_and_spectral_columns(list(df.columns))

    if dataset.has_mean_spectra_plot:
        # Plotted directly, unshifted: df's spectral values are Z-scores
        # (see the diagnostic below and decision_tree/config.py's module
        # docstring) - a negative or zero value is a normal, statistically
        # meaningful reading (this wavenumber's intensity at or below its
        # own population mean), not an instrument artifact, so there is
        # no principled floor to shift it to. See spectra_overview.py's
        # y-axis label for how this is made clear to the reader.
        plot_mean_spectra_by_oil(df, spectral_columns, output_dir / files["mean_spectra"])

    report_spectrum_minimum(df[spectral_columns])
    print(
        f"Raw column values are {dataset.negative_value_note}; negative values "
        "there are a real, physically meaningful reading, not an artifact - "
        "see the module docstring in decision_tree/config.py. They are left "
        "as-is here (unshifted); the modeling features derived from them are "
        "normalized separately below, via standardize_train_test."
    )
    df.info(memory_usage="deep")

    X = df.drop(columns=dataset.drop_columns).astype(float)
    y = df[dataset.target_column]

    # SCOPE: spectrum-level evaluation over a controlled sample cohort.
    # Each row here is one Raman spectrum; each physical sample (one oil
    # preparation, or one fried-chip batch) contributes ~20 spectra taken
    # at different spatial points, a standard approach for obtaining a
    # robust, averaged spectral signature per sample:
    #   - Oils.csv: 1000 rows = 50 physical oil samples (5 oil types x 10
    #     replicates; the "Oil_Type" column has exactly 50 unique values)
    #     x 20 spectra each.
    #   - Chips.csv and both NNLS-subtracted variants: 900 rows = 45
    #     physical chip batches (5 oil types x 9 frying cycles; the "Chips
    #     Type" column has exactly 45 unique values, and the subtracted
    #     datasets' "Replicate" column, 20 unique values, is the
    #     within-batch spatial-point index) x 20 spectra each.
    # `train_test_split`/`StratifiedKFold` below stratify on the 5-class
    # oil label and evaluate at the spectrum level, identically across
    # every dataset and pruning stage - which is what makes the
    # improvements attributed to NNLS correction and pruning directly
    # comparable to one another (see README's "Scope and Future Work").
    # A batch-independent (sample-level) split - via sklearn's
    # GroupShuffleSplit/GroupKFold, keyed on "Oil_Type"/"Chips Type" - is
    # a natural extension for validating generalization to entirely new
    # preparation batches as the sample cohort grows in future studies.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )

    # Column-wise (per-wavenumber) z-score normalization, fit on the
    # training split only, applied identically for every dataset this
    # pipeline runs on (pure oils, chips, and both NNLS-subtracted chip
    # variants). See data.standardize_train_test's docstring for the full
    # justification - equal per-mode weighting, why column- not row-wise,
    # why this is a units/consistency change only for a Decision Tree,
    # and the fit-on-train-only rationale - and
    # tests/test_scaling_invariance.py for the proof that it cannot
    # change predictions, accuracy, or feature-importance ranking, only
    # the numeric units of reported split thresholds.
    X_train, X_test = standardize_train_test(X_train_raw, X_test_raw)

    print("Shape of Training set : ", X_train.shape)
    print("Shape of test set : ", X_test.shape)
    print("Percentage of classes in training set:")
    print(y_train.value_counts(normalize=True))
    print("Percentage of classes in test set:")
    print(y_test.value_counts(normalize=True))

    label_encoder = LabelEncoder().fit(y)
    print(label_encoder.classes_)
    for index, name in enumerate(label_encoder.classes_):
        print(index, "->", name)

    default_train_perf, default_test_perf = _run_baseline_tree(
        X_train, y_train, X_test, y_test, output_dir, files, random_state
    )
    prepruned_model, feature_names, importances, prepruned_train_perf, prepruned_test_perf = (
        _run_pre_pruning(
            X_train, y_train, X_test, y_test, output_dir, files, label_encoder, random_state
        )
    )

    if dataset.has_four_feature_check:
        important_df = build_important_features_dataframe(
            df, feature_names, importances, dataset.target_column
        )
        important_df.to_csv(output_dir / files["important_features_csv"], index=False)
        important_df.info(memory_usage="deep")
        run_four_feature_check(important_df, output_dir, random_state, dataset.target_column)

    postpruned_train_perf, postpruned_test_perf = _run_post_pruning(
        X_train, y_train, X_test, y_test, prepruned_model, feature_names,
        output_dir, files, random_state,
    )

    print("Training performance comparison:")
    print(_compare(default_train_perf, prepruned_train_perf, postpruned_train_perf))
    print("Test set performance comparison:")
    print(_compare(default_test_perf, prepruned_test_perf, postpruned_test_perf))


def _run_baseline_tree(X_train, y_train, X_test, y_test, output_dir, files, random_state):
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    print(model)

    plot_confusion_matrix(
        model, X_train, y_train, OIL_CODES_SORTED,
        "Baseline (Default) Decision Tree — Training Set",
        output_dir / files["default_train_confusion"],
    )
    train_perf = classification_performance(model, X_train, y_train)
    print(train_perf)

    plot_confusion_matrix(
        model, X_test, y_test, OIL_CODES_SORTED,
        "Baseline (Default) Decision Tree — Test Set",
        output_dir / files["default_test_confusion"],
    )
    test_perf = classification_performance(model, X_test, y_test)
    print(test_perf)

    _print_sample_predictions(model, X_test, y_test)
    return train_perf, test_perf


def _run_pre_pruning(
    X_train, y_train, X_test, y_test, output_dir, files, label_encoder, random_state
):
    model, cv_results = pruning.grid_search_pre_pruned(X_train, y_train, random_state)
    print(cv_results.head())
    print(model)

    plot_confusion_matrix(
        model, X_train, y_train, OIL_CODES_SORTED, "Pre-Pruned Decision Tree — Training Set",
        output_dir / files["prepruned_train_confusion"],
    )
    train_perf = classification_performance(model, X_train, y_train)
    print(train_perf)

    plot_confusion_matrix(
        model, X_test, y_test, OIL_CODES_SORTED, "Pre-Pruned Decision Tree — Test Set",
        output_dir / files["prepruned_test_confusion"],
    )
    test_perf = classification_performance(model, X_test, y_test)
    print(test_perf)

    feature_names = list(X_train.columns)
    importances = model.feature_importances_

    reporting.plot_decision_tree_diagram(
        model, feature_names, OIL_CODES_SORTED, "Pre-Pruned Decision Tree",
        output_dir / files["prepruned_tree_plot"], figsize=(24, 12), fontsize=11,
    )

    class_mapping = dict(enumerate(label_encoder.classes_))
    report = reporting.export_tree_text_report(
        model, feature_names, output_dir / files["prepruned_text_report"],
        figsize=(14, 10), class_mapping=class_mapping,
    )
    print(report)

    print(int(np.sum(importances > 0)))
    reporting.plot_feature_importance(
        model, feature_names, output_dir / files["prepruned_feature_importance"],
        "Pre-pruned Decision Tree Feature Importances", "Relative Importance", figsize=(7, 4),
    )
    for name, importance in zip(feature_names, importances):
        if importance > 0:
            print(f"{name} : {importance:.4f}")

    return model, feature_names, importances, train_perf, test_perf


def _run_post_pruning(
    X_train, y_train, X_test, y_test, prepruned_model, feature_names,
    output_dir, files, random_state,
):
    ccp_alphas, impurities = pruning.cost_complexity_pruning_path(X_train, y_train, random_state)
    print(pd.DataFrame({"ccp_alphas": ccp_alphas, "impurities": impurities}))
    print(prepruned_model.get_n_leaves())
    print(prepruned_model.get_depth())
    for alpha in ccp_alphas:
        # class_weight="balanced" here to match fit_pruned_tree_sequence
        # and the final selected model below - this loop is a depth/leaf
        # diagnostic only (its trees are never kept), but should still
        # describe the same tree the rest of this stage actually uses.
        probe = DecisionTreeClassifier(
            random_state=random_state, ccp_alpha=alpha, class_weight="balanced"
        )
        probe.fit(X_train, y_train)
        print(f"alpha={alpha:.6f}", "depth=", probe.get_depth(), "leaves=", probe.get_n_leaves())

    trees = pruning.fit_pruned_tree_sequence(X_train, y_train, ccp_alphas, random_state)
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            trees[-1].tree_.node_count, ccp_alphas[-1]
        )
    )

    model, cv_table = pruning.grid_search_post_pruned(X_train, y_train, ccp_alphas, random_state)
    print(cv_table.head())

    # Drop the last alpha for the diagnostic recall/accuracy curves only -
    # it always collapses the tree to an uninformative single root node.
    trees = trees[:-1]
    pruning.weighted_recall_curve(trees, X_train, y_train)
    pruning.weighted_recall_curve(trees, X_test, y_test)
    pruning.accuracy_curve(trees, X_train, y_train)
    pruning.accuracy_curve(trees, X_test, y_test)

    plot_confusion_matrix(
        model, X_train, y_train, OIL_CODES_SORTED, "Post-Pruned Decision Tree — Training Set",
        output_dir / files["postpruned_train_confusion"],
    )
    train_perf = classification_performance(model, X_train, y_train)
    print(train_perf)

    plot_confusion_matrix(
        model, X_test, y_test, OIL_CODES_SORTED, "Post-Pruned Decision Tree — Test Set",
        output_dir / files["postpruned_test_confusion"],
    )
    test_perf = classification_performance(model, X_test, y_test)
    print(test_perf)

    reporting.plot_decision_tree_diagram(
        model, feature_names, OIL_CODES_SORTED, "Post-Pruned Decision Tree",
        output_dir / files["postpruned_tree_plot"], figsize=(22, 12), fontsize=10,
        arrow_linewidth=1,
    )

    report = reporting.export_tree_text_report(
        model, feature_names, output_dir / files["postpruned_text_report"], figsize=(14, 8),
    )
    print(report)

    importances = model.feature_importances_
    print(int(np.sum(importances > 0)))
    reporting.plot_feature_importance(
        model, feature_names, output_dir / files["postpruned_feature_importance"],
        "Post-pruned Decision Tree Feature Importances", "Feature Importance",
        figsize=(8, 4), show_value_labels=True,
    )

    used = importances > 0
    importance_table = pd.DataFrame(
        {"Wavenumber": np.array(feature_names)[used], "Importance": importances[used]}
    ).sort_values("Importance", ascending=False)
    print(importance_table)

    return train_perf, test_perf


def _print_sample_predictions(model, X_test, y_test, n_samples: int = 3) -> None:
    sample_rows = X_test.iloc[:n_samples]
    predictions = model.predict(sample_rows)
    for i in range(n_samples):
        print(f"---test case {i + 1} ---")
        print("Input Features")
        print(sample_rows.iloc[i])
        print("True Label:", y_test.iloc[i])
        print("Predicted Label:", predictions[i])
        print()


def _compare(default_perf, prepruned_perf, postpruned_perf) -> pd.DataFrame:
    comparison = pd.concat([default_perf.T, prepruned_perf.T, postpruned_perf.T], axis=1)
    comparison.columns = [
        "Decision Tree (sklearn default)",
        "Decision Tree (Pre-Pruning)",
        "Decision Tree (Post-Pruning)",
    ]
    return comparison
