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
from sklearn.preprocessing import StandardScaler


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
    print("Row index:", row_idx)
    print("Wavenumber:", col_idx)
    return min_value, row_idx, col_idx


def standardize_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Column-wise (per-wavenumber) z-score, fit on the training split only.

    Applied identically, by the Decision Tree pipeline, to every dataset
    in this project (pure oils, chips, and both NNLS-subtracted chip
    variants).

    Why normalize at all
    ---------------------
    Each wavenumber column is a distinct Raman-active vibrational mode
    with its own intrinsic scattering cross-section; cross-sections
    differ by orders of magnitude between modes for physical reasons
    that have nothing to do with how well a wavenumber discriminates
    between oils (e.g. C-H stretching, ~2850-2950 cm^-1, is
    characteristically far more Raman-intense than many
    fingerprint-region skeletal modes below 1500 cm^-1). Column-wise
    z-scoring ("autoscaling" in chemometrics) puts every vibrational
    mode on equal statistical footing - the same reasoning that makes
    K-Means/t-SNE distances meaningful (see ``clustering/scaling.py``).

    Why it is applied here too, for Decision Trees, which don't need it
    ---------------------------------------------------------------------
    A Decision Tree's splits are threshold cuts on one feature at a
    time; any per-feature affine rescaling (which z-scoring is)
    preserves each feature's sample rank order, so it cannot change
    which side of a split any sample falls on. Accuracy, confusion
    matrices, and feature-importance *ranking* are therefore
    mathematically guaranteed identical whether or not this function is
    used - see ``tests/test_scaling_invariance.py`` for a direct proof.
    It is applied anyway so that the two datasets that arrive
    unnormalized (the NNLS-subtracted chip variants; see the module
    docstring in ``decision_tree/config.py``) are treated identically to
    the two that arrive pre-normalized (pure oils, chips), and so every
    reported split threshold and feature-importance value in this
    project is expressed in the same, comparable (z-score) unit.

    Why column-wise, not row-wise (per-spectrum)
    -----------------------------------------------
    Row-wise normalization (e.g. Standard Normal Variate) corrects a
    different problem - multiplicative drift between *acquisitions*
    (laser power, exposure, sample positioning) - by rescaling each
    spectrum to its own mean/std. That is not attempted anywhere in this
    project; this function only ever normalizes across the sample
    population, per wavenumber.

    Why fit on ``X_train`` only
    ------------------------------
    Standard practice for any train/test-based transform: fitting on the
    full dataset (train + test) would let the held-out test set's
    distribution influence the transform applied to training data. For a
    Decision Tree this leakage happens to be provably harmless (see
    above), but there is no reason to rely on that fact when the
    correct-by-construction alternative costs nothing.

    A caveat worth disclosing: the pure-oils and chips CSVs were already
    column-standardized *before* they entered this repository (see
    ``decision_tree/config.py``'s ``_ZSCORE_NOTE``) - this function
    re-standardizes their *training-split* values on top of that, which
    is an (almost, but not exactly, identity) second pass rather than
    the first true normalization for those two datasets. The original
    upstream normalization cannot be independently re-derived or
    verified from raw, pre-normalization intensities, because those raw
    values are not available in this repository.

    A second, related question worth answering explicitly: this function
    is called *once* on the whole training split, before
    ``decision_tree/pruning.py`` runs its internal 5-fold
    ``GridSearchCV`` for hyperparameter selection - the scaler is not
    refit inside each CV fold (e.g. via a ``sklearn.pipeline.Pipeline``).
    In general that is a mild form of preprocessing leakage: a fold's
    held-out validation rows influence the mean/std used to scale that
    same fold's training rows. For a Decision Tree it remains harmless
    for the same reason as above: within any one CV fold, every row -
    train and validation alike - is transformed by the *same fixed*
    per-feature affine map, and an affine map preserves each feature's
    rank order across any subset of rows regardless of which rows its
    own parameters were estimated from. So the split chosen at every
    node, and therefore the selected hyperparameters and CV scores, are
    identical to what a per-fold-refit scaler (or no scaler at all) would
    have produced. This only holds because the estimator here is a
    Decision Tree; it would not generalize to a scale-sensitive model
    (e.g. SVM, k-NN, logistic regression), which would need the scaler
    placed inside a ``Pipeline`` and refit per fold.
    """
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return train_scaled, test_scaled
