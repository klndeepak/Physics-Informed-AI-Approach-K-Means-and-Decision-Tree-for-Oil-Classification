#!/usr/bin/env python3
"""Decision Tree classification of the paper-and-potato-subtracted chip spectra.

Reproduces ``Decision Tree_Chips - Paper and Potato Subtracted.ipynb``:
the same workflow again, on chip spectra with both the paper and potato
matrix contributions removed via NNLS - the most heavily
physics-corrected of the four datasets. Figures and reports are written
under ``Images/Chips-Paper and Potato Subtracted/``.

Usage::

    python scripts/run_decision_tree_chips_paper_and_potato_subtracted.py
"""

from raman_analysis.decision_tree import config, pipeline


def main() -> None:
    pipeline.run(config.CHIPS_PAPER_AND_POTATO_SUBTRACTED)


if __name__ == "__main__":
    main()
