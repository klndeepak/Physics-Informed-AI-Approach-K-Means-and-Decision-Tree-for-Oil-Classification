#!/usr/bin/env python3
"""Decision Tree classification of the pure-oil Raman spectra.

Reproduces ``Decision Tree_Oils.ipynb``: baseline, pre-pruned, and
post-pruned trees, plus a check that only four wavenumbers are needed
for perfect classification. Figures and reports are written under
``Images/Oils/``.

Usage::

    python scripts/run_decision_tree_oils.py
"""

from raman_analysis.decision_tree import config, pipeline


def main() -> None:
    pipeline.run(config.OILS)


if __name__ == "__main__":
    main()
