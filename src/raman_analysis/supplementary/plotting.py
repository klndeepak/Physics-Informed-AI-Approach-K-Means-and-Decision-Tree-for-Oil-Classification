"""PNG summaries drawn exclusively from generated CSV tables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATASET_LABELS = {
    "oils": "Oils",
    "chips": "Chips",
    "chips_paper_subtracted": "Chips: paper subtracted",
    "chips_paper_and_potato_subtracted": "Chips: paper and potato subtracted",
}
MODEL_LABELS = {
    "baseline": "Baseline",
    "pre_pruned": "Pre-pruned",
    "post_pruned": "Post-pruned",
}


def _display_labels(table):
    table = table.copy()
    if "dataset" in table:
        table["dataset"] = table["dataset"].map(DATASET_LABELS)
    if "model" in table:
        table["model"] = table["model"].map(MODEL_LABELS)
    return table


def _finish(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def model_performance(table: pd.DataFrame, path):
    test = _display_labels(table[table["split"] == "test"])
    plt.figure(figsize=(12, 5))
    sns.barplot(data=test, x="dataset", y="f1_weighted", hue="model")
    plt.ylim(0, 1.05)
    plt.ylabel("Weighted F1 score")
    plt.xlabel("Dataset")
    plt.xticks(rotation=10, ha="right")
    _finish(path)


def model_robustness(table: pd.DataFrame, path):
    table = _display_labels(table)
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=table, x="dataset", y="f1_weighted", hue="model")
    plt.ylim(0, 1.05)
    plt.ylabel("Weighted F1 score across splits")
    plt.xlabel("Dataset")
    plt.xticks(rotation=10, ha="right")
    _finish(path)


def kmeans_stability(table: pd.DataFrame, path):
    table = _display_labels(table)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.lineplot(
        data=table, x="seed", y="ari_vs_reference", hue="dataset", marker="o", ax=axes[0]
    )
    sns.lineplot(
        data=table, x="seed", y="silhouette_score", hue="dataset", marker="o", ax=axes[1]
    )
    axes[0].set_ylabel("ARI versus reference clustering")
    axes[1].set_ylabel("Silhouette score")
    for axis in axes:
        axis.set_xlabel("Random seed")
    _finish(path)


def pruning_path(table: pd.DataFrame, path):
    table = _display_labels(table).melt(
        id_vars=["dataset", "ccp_alpha"],
        value_vars=["train_accuracy", "test_accuracy"],
        var_name="split",
        value_name="accuracy",
    )
    table["split"] = table["split"].map(
        {"train_accuracy": "Train", "test_accuracy": "Test"}
    )
    grid = sns.FacetGrid(
        table, col="dataset", col_wrap=2, hue="split", height=3.4, sharex=False
    )
    grid.map_dataframe(sns.lineplot, x="ccp_alpha", y="accuracy")
    grid.set_axis_labels("Cost-complexity alpha", "Accuracy")
    grid.add_legend()
    grid.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(grid.figure)


def feature_stability(table: pd.DataFrame, path):
    filtered = table[(table["model"] != "baseline") & (table["selection_count"] > 0)]
    ranks = (
        filtered.groupby(["dataset", "wavenumber"], as_index=False)
        .agg(selection_rate=("selection_rate", "max"), mean_importance=("mean_importance", "max"))
        .sort_values(
            ["dataset", "selection_rate", "mean_importance"], ascending=[True, False, False]
        )
        .groupby("dataset", as_index=False)
        .head(8)
    )
    top = filtered.merge(ranks[["dataset", "wavenumber"]], on=["dataset", "wavenumber"])
    top = _display_labels(top)
    top["wavenumber"] = top["wavenumber"].astype(str)
    grid = sns.catplot(
        data=top,
        x="selection_rate",
        y="wavenumber",
        hue="model",
        col="dataset",
        col_wrap=2,
        kind="bar",
        height=4,
        aspect=1.25,
        sharey=False,
    )
    grid.set(xlim=(0, 1.05))
    grid.set_axis_labels("Selection rate across splits", "Wavenumber")
    grid.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(grid.figure)
