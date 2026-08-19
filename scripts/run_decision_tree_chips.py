#!/usr/bin/env python3
"""Decision Tree classification of the raw fried-chip Raman spectra.

Reproduces ``Decision Tree_Chips.ipynb``: baseline, pre-pruned, and
post-pruned trees over the uncorrected chip spectra (oils embedded in
the potato/paper matrix). Figures and reports are written under
``Images/Chips/``.

Usage::

    python scripts/run_decision_tree_chips.py
"""

from raman_analysis.decision_tree import config, pipeline


def main() -> None:
    pipeline.run(config.CHIPS)


if __name__ == "__main__":
    main()
