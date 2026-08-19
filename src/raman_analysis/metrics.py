"""Classification metrics shared by every Decision Tree model.

Both helpers here are dataset-agnostic: they take a fitted classifier
plus a feature/target pair and report weighted-average metrics, since
every dataset in this project is a 5-class (oil type) problem where each
class matters equally regardless of how many samples it has.
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


def classification_performance(model, predictors, target) -> pd.DataFrame:
    """One-row DataFrame of weighted accuracy/recall/precision/F1."""
    pred = model.predict(predictors)
    return pd.DataFrame(
        {
            "Accuracy": accuracy_score(target, pred),
            "Recall": recall_score(target, pred, average="weighted"),
            "Precision": precision_score(target, pred, average="weighted"),
            "F1": f1_score(target, pred, average="weighted"),
        },
        index=[0],
    )


def plot_confusion_matrix(
    model,
    predictors,
    target,
    class_labels: list[str],
    title: str,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Draw, save, and close a confusion-matrix heatmap.

    Each cell is annotated with its raw count and its share of the total.
    ``class_labels`` must list the classes in the same order sklearn's
    ``confusion_matrix`` uses (alphabetical), since it is only used to
    relabel the tick marks - the matrix itself is computed from
    ``target``/predictions directly.
    """
    y_pred = model.predict(predictors)
    n_classes = len(class_labels)
    cm = confusion_matrix(target, y_pred)
    total = cm.flatten().sum()
    cell_labels = np.asarray(
        [f"{count:0.0f}\n{count / total:.2%}" for count in cm.flatten()]
    ).reshape(n_classes, n_classes)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=cell_labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    ax = plt.gca()
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_title(title)

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
