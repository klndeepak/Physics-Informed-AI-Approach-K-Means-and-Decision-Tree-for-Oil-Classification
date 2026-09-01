"""Reduced-feature reproduction check (pure oils dataset only).

The pre-pruned tree for the pure oils turns out to use only four
wavenumbers. This step re-trains a small tree on *just* those four
columns to confirm they alone are sufficient for perfect classification
- the headline result of the "physics-informed" side of the study
(perfect separation from 4 of 1866 available variables). It is specific
to the oils dataset, which is why it lives outside the shared
``decision_tree/pipeline.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .reporting import export_tree_text_report, plot_decision_tree_diagram
from ..data import standardize_train_test
from ..paths import ensure_dir

# Hyperparameters mirror the winning pre-pruned model, since this step
# exists to confirm that model's own feature choices, not to re-tune.
FOUR_FEATURE_TREE_PARAMS = dict(
    class_weight="balanced", max_depth=4, max_leaf_nodes=50, min_samples_split=10
)


def build_important_features_dataframe(
    df_oils: pd.DataFrame,
    feature_names: list[str],
    importances: np.ndarray,
    target_column: str = "Oil",
) -> pd.DataFrame:
    """Subset ``df_oils`` to the target column plus tree-used wavenumbers."""
    used_features = [name for name, imp in zip(feature_names, importances) if imp > 0]
    return df_oils[[target_column] + used_features].copy()


def _print_performance(label: str, y_true, y_pred) -> None:
    print(f"\n{label} Performance")
    print("-" * (len(label) + 12))
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred, average='weighted'):.4f}")


def _plot_four_feature_confusion_matrix(y_test, y_test_pred, class_labels, out_path):
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Oil")
    plt.ylabel("Actual Oil")
    plt.title("Decision Tree Using Only 4 Wavenumbers")
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def run_four_feature_check(
    df_features: pd.DataFrame,
    output_dir: Path,
    random_state: int,
    target_column: str = "Oil",
) -> DecisionTreeClassifier:
    """Fit and report a tree restricted to ``df_features``'s wavenumbers.

    ``df_features`` is expected to already be limited to the target
    column plus the handful of wavenumbers the pre-pruned tree actually
    split on (see :func:`build_important_features_dataframe`).
    """
    ensure_dir(output_dir)
    selected_wavenumbers = [c for c in df_features.columns if c != target_column]
    X = df_features[selected_wavenumbers].astype(float)
    y = df_features[target_column]

    # Same spectrum-level evaluation scope as the main pipeline's split
    # (see decision_tree/pipeline.py's "SCOPE" note): stratified by class,
    # consistent with every other split in this study.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )

    # Same column-wise z-score policy as the main pipeline (see
    # data.standardize_train_test's docstring), fit independently here
    # because this step retrains on a different feature subset - just
    # the tree-selected wavenumbers - than the full-feature model above.
    # As there, this is a units/consistency choice, not a correctness
    # requirement: restricting to 4 already-selected features and
    # re-fitting a Decision Tree on them is unaffected by their scale.
    X_train, X_test = standardize_train_test(X_train_raw, X_test_raw)

    model = DecisionTreeClassifier(random_state=random_state, **FOUR_FEATURE_TREE_PARAMS)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Using ONLY these 4 wavenumbers:")
    print(selected_wavenumbers)
    _print_performance("Training", y_train, y_train_pred)
    _print_performance("Test", y_test, y_test_pred)

    class_labels = sorted(y.unique())
    _plot_four_feature_confusion_matrix(
        y_test, y_test_pred, class_labels,
        output_dir / "confusion_matrix_test_4_wavenumbers_oils.jpg",
    )

    plot_decision_tree_diagram(
        model,
        feature_names=selected_wavenumbers,
        class_names=list(model.classes_),
        title="Decision Tree Using Only 4 Wavenumbers",
        out_path=output_dir / "FourFeature_Decision_Tree_oils.jpg",
        figsize=(16, 8),
        fontsize=11,
        arrow_linewidth=1.2,
    )

    class_mapping = dict(enumerate(model.classes_))
    export_tree_text_report(
        model,
        feature_names=selected_wavenumbers,
        out_path=output_dir / "FourFeature_Decision_Tree_Text_Report_Oils.jpg",
        figsize=(10, 6),
        class_mapping=class_mapping,
        rules_header="Decision Tree Rules (4 Wavenumbers)",
    )

    results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_test_pred})
    results["Correct"] = results["Actual"] == results["Predicted"]
    print(results["Correct"].value_counts())
    print("\nNumber of incorrect predictions:", (~results["Correct"]).sum())
    print("Test samples:", len(results))
    print("Test accuracy:", results["Correct"].mean())

    return model
