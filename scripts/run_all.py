#!/usr/bin/env python3
"""Run every analysis in the project, in one command.

Convenience wrapper around the six individual scripts in this
directory - useful for regenerating every figure under ``Images/``
from scratch. For a single analysis, run its script directly instead
(e.g. ``python scripts/run_decision_tree_oils.py``); this full sweep
takes a while, most of it in the t-SNE and Decision Tree grid searches.

Usage::

    python scripts/run_all.py
"""

from raman_analysis.clustering import config as clustering_config
from raman_analysis.clustering import pipeline as clustering_pipeline
from raman_analysis.clustering import compare as clustering_compare
from raman_analysis.clustering.requirements_report import write_environment_report
from raman_analysis.decision_tree import config as decision_tree_config
from raman_analysis.decision_tree import pipeline as decision_tree_pipeline


def main() -> None:
    print("=== K-Means / t-SNE: Oils ===")
    oils_result = clustering_pipeline.run(clustering_config.OILS)
    write_environment_report(clustering_config.CLUSTER_OUTPUT_DIR / "requirements.md")

    print("=== K-Means / t-SNE: Chips ===")
    chips_result = clustering_pipeline.run(clustering_config.CHIPS)

    print("=== K-Means / t-SNE: Oils vs. Chips comparison ===")
    output_dir = clustering_config.CLUSTER_OUTPUT_DIR
    clustering_compare.plot_combined_wcss(
        oils_result.wcss, chips_result.wcss, output_dir / "K-Clusters-Combined.jpg"
    )
    clustering_compare.plot_combined_silhouette(
        oils_result.silhouette_scores, chips_result.silhouette_scores,
        output_dir / "silhouette_combined.jpg",
    )

    print("=== Decision Tree: Oils ===")
    decision_tree_pipeline.run(decision_tree_config.OILS)

    print("=== Decision Tree: Chips ===")
    decision_tree_pipeline.run(decision_tree_config.CHIPS)

    print("=== Decision Tree: Chips (Paper Subtracted) ===")
    decision_tree_pipeline.run(decision_tree_config.CHIPS_PAPER_SUBTRACTED)

    print("=== Decision Tree: Chips (Paper and Potato Subtracted) ===")
    decision_tree_pipeline.run(decision_tree_config.CHIPS_PAPER_AND_POTATO_SUBTRACTED)


if __name__ == "__main__":
    main()
