"""Figures and text reports rendered from a fitted Decision Tree.

Every notebook additionally drew a second, unlabeled "quick look" tree
diagram and a top-25 feature-importance chart right after each of these
plots. Neither was ever written to disk (no ``savefig`` call, only
``plt.show()``) - they were interactive-only duplicates of the figures
below, produced for on-screen exploration while writing the notebook.
Since they left no artifact behind, dropping them changes nothing about
the project's actual results; keeping them here would just make a
non-interactive script re-render the same tree/plot for no saved output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree as sklearn_tree


def plot_decision_tree_diagram(
    model,
    feature_names: list[str],
    class_names: list[str],
    title: str,
    out_path: Path,
    figsize: tuple[float, float],
    fontsize: int,
    arrow_linewidth: float = 1.2,
    dpi: int = 600,
) -> None:
    """Render and save a full sklearn tree diagram (nodes, splits, gini)."""
    plt.figure(figsize=figsize)
    artists = sklearn_tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=fontsize,
        node_ids=False,
    )
    for artist in artists:
        arrow = artist.arrow_patch
        if arrow is not None:
            arrow.set_edgecolor("black")
            arrow.set_linewidth(arrow_linewidth)

    plt.tight_layout()
    plt.title(title)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def export_tree_text_report(
    model,
    feature_names: list[str],
    out_path: Path,
    figsize: tuple[float, float],
    class_mapping: dict[int, str] | None = None,
    rules_header: str = "Decision Tree Rules",
    dpi: int = 600,
) -> str:
    """Render ``sklearn.tree.export_text`` as a saved monospace figure.

    When ``class_mapping`` is given (index -> class name), a "Class
    Mapping" section is prepended above the rules so the leaf-node class
    indices in the report are human-readable. Returns the raw rules text
    (without the mapping header) for console logging.
    """
    report = sklearn_tree.export_text(
        model, feature_names=feature_names, show_weights=True
    )

    if class_mapping is not None:
        mapping_text = "\n".join(
            f"{index}: {name}" for index, name in class_mapping.items()
        )
        full_report = (
            "Class Mapping\n"
            "-----------------\n"
            f"{mapping_text}\n\n"
            f"{rules_header}\n"
            "-----------------\n"
            f"{report}"
        )
    else:
        full_report = report

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(
        0.01, 0.99, full_report,
        transform=ax.transAxes, family="monospace", fontsize=10, va="top",
    )
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return report


def plot_feature_importance(
    model,
    feature_names: list[str],
    out_path: Path,
    title: str,
    xlabel: str,
    figsize: tuple[float, float],
    show_value_labels: bool = False,
    dpi: int = 600,
) -> None:
    """Horizontal bar chart of the nonzero Gini feature importances.

    Only wavenumbers actually used by a split get a nonzero importance,
    so (as in the original notebooks) the zero-importance majority of
    features is filtered out before plotting.
    """
    importances = model.feature_importances_
    used = importances > 0
    used_features = np.array(feature_names)[used]
    used_importances = importances[used]

    order = np.argsort(used_importances)
    sorted_features = used_features[order]
    sorted_importances = used_importances[order]

    plt.figure(figsize=figsize)
    plt.barh(sorted_features, sorted_importances)
    plt.xlabel(xlabel)
    plt.ylabel("Wavenumber (cm⁻¹)")
    plt.title(title)

    if show_value_labels:
        for row, value in enumerate(sorted_importances):
            plt.text(value, row, f"{value:.3f}", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
