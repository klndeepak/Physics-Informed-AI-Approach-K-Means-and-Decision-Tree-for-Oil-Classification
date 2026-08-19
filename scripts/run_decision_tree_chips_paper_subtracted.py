#!/usr/bin/env python3
"""Decision Tree classification of the paper-subtracted chip spectra.

Reproduces ``Decision Tree_Chips - Paper Subtracted.ipynb``: the same
baseline/pre-pruned/post-pruned workflow as the raw chip spectra, run
instead on the NNLS paper-corrected spectra, to test whether removing
the paper background's spectral contribution improves class separation.
Figures and reports are written under ``Images/Chips-Paper Subtracted/``.

Usage::

    python scripts/run_decision_tree_chips_paper_subtracted.py
"""

from raman_analysis.decision_tree import config, pipeline


def main() -> None:
    pipeline.run(config.CHIPS_PAPER_SUBTRACTED)


if __name__ == "__main__":
    main()
