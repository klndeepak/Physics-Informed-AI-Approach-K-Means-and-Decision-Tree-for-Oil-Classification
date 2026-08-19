"""Physics-informed ML analysis of Raman spectra for edible oils and chips.

This package is the modular, script-friendly replacement for the original
Jupyter notebooks. It exposes two analysis families:

- ``raman_analysis.clustering``: K-Means / t-SNE exploration of spectral
  structure (elbow, silhouette, cluster profiles).
- ``raman_analysis.decision_tree``: interpretable Decision Tree
  classification (baseline, pre-pruned, post-pruned, feature importance).

Both families are driven by the dataset configs in
``raman_analysis.config`` so the same code runs against any of the four
spectral datasets (Oils, Chips, Chips-Paper-Subtracted,
Chips-Paper-and-Potato-Subtracted).
"""

import pandas as pd

__all__ = ["__version__"]

__version__ = "1.0.0"

# Every DataFrame this project prints (metrics, cross-tabs, cv results)
# is meant to be read in full in a console. Without this, pandas wraps
# or truncates wide tables based on detected terminal width - fine in a
# notebook's HTML rendering, but it silently hides columns when printed
# from a plain script.
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
