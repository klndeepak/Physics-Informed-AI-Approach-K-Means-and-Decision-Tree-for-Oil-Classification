"""Loading and column bookkeeping shared by every spectral dataset.

Each of the four source CSVs stores a handful of metadata columns (oil
type, chip stage, replicate, ...) alongside one column per Raman
wavenumber. The wavenumber columns are not always stored in the same
order - ``Chips.csv`` happens to be saved high-to-low, the other three
low-to-high - so every pipeline in this project loads through
:func:`load_spectral_dataset`, which always returns metadata columns
first (in their original order) followed by wavenumber columns sorted
ascending. That single rule reproduces what each original notebook did
in its own, dataset-specific way (an explicit ascending sort for the
Decision Tree notebooks, a manual column reversal for the Chips K-Means
notebook, and a no-op for the already-ascending Oils K-Means notebook).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def is_wavenumber_column(column_name: str) -> bool:
    """Whether a column name parses as a number (i.e. is a wavenumber)."""
    try:
        float(column_name)
    except ValueError:
        return False
    return True


def split_meta_and_spectral_columns(
    columns: list[str],
) -> tuple[list[str], list[str]]:
    """Split ``columns`` into (metadata columns, wavenumber columns).

    Metadata columns keep their original relative order; wavenumber
    columns are returned sorted ascending by their numeric value.
    """
    meta_columns = [c for c in columns if not is_wavenumber_column(c)]
    spectral_columns = sorted(
        (c for c in columns if is_wavenumber_column(c)), key=float
    )
    return meta_columns, spectral_columns


def load_spectral_dataset(
    csv_path: Path, drop_unnamed_index: bool = False
) -> pd.DataFrame:
    """Load a Raman spectral CSV with a canonical column order.

    Parameters
    ----------
    csv_path:
        Path to the dataset CSV.
    drop_unnamed_index:
        ``Oils.csv`` and ``Chips.csv`` were exported with an extra,
        unlabeled index column (read back by pandas as ``"Unnamed: 0"``);
        the two "Subtracted" datasets were not. Set this to ``True`` for
        the former.
    """
    df = pd.read_csv(csv_path)
    if drop_unnamed_index:
        df = df.drop(columns=["Unnamed: 0"])

    meta_columns, spectral_columns = split_meta_and_spectral_columns(
        list(df.columns)
    )
    return df[meta_columns + spectral_columns]


def report_spectrum_minimum(
    spectra: pd.DataFrame,
) -> tuple[float, object, str]:
    """Print and return the most negative value in a spectral block.

    Mirrors the diagnostic printout every notebook ran before deciding
    whether (and how) to handle negative intensities: the minimum value,
    which sample row it occurs in, and which wavenumber column.
    """
    min_value = spectra.min().min()
    row_idx, col_idx = (spectra == min_value).stack().idxmax()

    print("Most negative value:", min_value)
    print("Most negative value:", min_value)
    print("Row index:", row_idx)
    print("Wavenumber:", col_idx)
    return min_value, row_idx, col_idx


def shift_to_nonnegative(
    df: pd.DataFrame, spectral_columns: list[str], min_value: float
) -> pd.DataFrame:
    """Add ``abs(min_value)`` to every spectral column so the minimum is 0.

    Used by the clustering pipeline, where distance-based algorithms
    (K-Means, t-SNE) benefit from a consistent, non-negative scale.
    The Decision Tree pipeline does not shift its data - see
    :func:`report_spectrum_minimum`'s callers for why.
    """
    shifted = df.copy()
    shifted[spectral_columns] = shifted[spectral_columns] + abs(min_value)
    print("New minimum:", shifted[spectral_columns].min().min())
    return shifted
