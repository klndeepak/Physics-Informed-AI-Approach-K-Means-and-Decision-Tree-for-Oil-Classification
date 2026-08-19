"""Generate all additional CSV tables and PNG figures."""

from __future__ import annotations

import pandas as pd

from . import clustering, decision_tree, plotting
from ..clustering import config as clustering_config
from ..decision_tree import config as decision_tree_config
from ..paths import PROJECT_ROOT, ensure_dir

OUTPUT_DIR = PROJECT_ROOT / "supplementary"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"


def _save_tables(tables: dict[str, pd.DataFrame]):
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False)


def _render(tables):
    ensure_dir(FIGURE_DIR)
    plotting.model_performance(
        tables["model_metrics_summary"], FIGURE_DIR / "model_performance_summary.png"
    )
    plotting.model_robustness(
        tables["model_robustness"], FIGURE_DIR / "model_robustness.png"
    )
    plotting.kmeans_stability(
        tables["kmeans_stability"], FIGURE_DIR / "kmeans_stability.png"
    )
    plotting.pruning_path(tables["pruning_results"], FIGURE_DIR / "pruning_path.png")
    plotting.feature_stability(
        tables["feature_stability"], FIGURE_DIR / "feature_stability.png"
    )


def render_saved_tables():
    """Regenerate PNG files from the existing CSV tables."""
    tables = {path.stem: pd.read_csv(path) for path in TABLE_DIR.glob("*.csv")}
    _render(tables)


def run():
    ensure_dir(TABLE_DIR)
    ensure_dir(FIGURE_DIR)

    tree_tables = decision_tree.collect(decision_tree_config.DATASETS.values())
    cluster_tables = clustering.collect(clustering_config.DATASETS.values())
    tables = {**tree_tables, **cluster_tables}
    _save_tables(tables)
    _render(tables)
    print(f"Wrote additional results to {OUTPUT_DIR}")
