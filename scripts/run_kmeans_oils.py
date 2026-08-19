#!/usr/bin/env python3
"""K-Means / t-SNE clustering of the pure-oil Raman spectra.

Reproduces ``K-Means Clusters_Oils.ipynb``: elbow and silhouette
analysis, 2D/3D t-SNE visualization, K-Means clustering (K=5), and
per-cluster spectral profiles. Also records the exact library versions
used, in ``Images/K-Means Clusters/requirements.md``.

Run ``run_kmeans_comparison.py`` afterward (or instead, if you only want
the cross-dataset charts) to additionally produce the oils-vs-chips
comparison figures, which need both datasets' results.

Usage::

    python scripts/run_kmeans_oils.py
"""

from raman_analysis.clustering import config, pipeline
from raman_analysis.clustering.requirements_report import write_environment_report


def main() -> None:
    pipeline.run(config.OILS)
    write_environment_report(config.CLUSTER_OUTPUT_DIR / "requirements.md")


if __name__ == "__main__":
    main()
