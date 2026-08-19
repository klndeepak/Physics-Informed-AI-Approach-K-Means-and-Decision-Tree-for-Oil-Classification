"""Filesystem locations shared by every pipeline.

The original notebooks hard-coded Windows-style paths such as
``r"Images\\Oils\\confusion_matrix_train_prepruned_oil.jpg"``. On any other
platform a backslash is just a character, not a path separator, so those
strings silently wrote one oddly-named file into the working directory
instead of into ``Images/Oils/``. Every path in this project is built with
:mod:`pathlib` instead, so it is correct on Windows, macOS, and Linux alike.
"""

from pathlib import Path

# Repository root: two levels up from this file (src/raman_analysis/paths.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# The canonical notebooks and datasets live together at the repository root.
# Keeping the production pipeline pointed at the same CSVs prevents two copies
# of the scientific inputs from drifting apart.
DATA_DIR = PROJECT_ROOT
IMAGES_DIR = PROJECT_ROOT / "Images"


def ensure_dir(directory: Path) -> Path:
    """Create ``directory`` (and any parents) if it does not exist yet.

    Returns the same path so this can be chained inline, e.g.
    ``out_file = ensure_dir(some_dir) / "plot.jpg"``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory
