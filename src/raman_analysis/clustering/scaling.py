"""Column-wise (per-wavenumber) z-score scaling - the step that makes
K-Means and t-SNE's Euclidean distances meaningful on Raman spectra.

Why normalize at all
---------------------
Each wavenumber column is a distinct Raman-active vibrational mode with
its own intrinsic scattering cross-section; cross-sections differ by
orders of magnitude between modes for physical reasons unrelated to how
well a wavenumber discriminates between oils (e.g. C-H stretching,
~2850-2950 cm^-1, is characteristically far more Raman-intense than many
fingerprint-region skeletal modes below 1500 cm^-1). Without scaling,
Euclidean distance - and therefore every K-Means cluster and every t-SNE
neighborhood - would be dominated by whichever few wavenumbers happen to
have the largest raw variance, rather than by which wavenumbers actually
separate the oils.

Why column-wise ("autoscaling"), not row-wise (per-spectrum)
-----------------------------------------------------------------
This scales each wavenumber (feature) to unit variance across the
sample population, putting every vibrational mode on equal footing
regardless of its physical cross-section. This is a different operation
from row-wise normalization (e.g. Standard Normal Variate), which
corrects a different problem - multiplicative drift between
*acquisitions* (laser power, exposure, sample positioning) - by
rescaling each spectrum to its own mean/std; that is not attempted
anywhere in this project.

A caveat worth disclosing: the pure-oils and chips CSVs this function
runs on were already column-standardized *before* they entered this
repository (see ``raman_analysis.decision_tree.config``'s
``_ZSCORE_NOTE``, and the README's Datasets section) - this call is
therefore a defensive, explicit re-application on top of an
already-(near-)standardized input for those two datasets, kept so the
pipeline does not silently depend on that upstream fact and stays
correct if it is ever pointed at a dataset that is not pre-scaled.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """Z-score scale every column of ``X`` (population std, matching
    ``StandardScaler``'s convention) and return it as a DataFrame with
    the original column names. Unlike ``data.standardize_train_test``,
    this fits on all of ``X`` at once - appropriate here because K-Means
    and t-SNE are unsupervised (there is no held-out test set whose
    distribution this could leak from).
    """
    scaled = StandardScaler().fit_transform(X)
    return pd.DataFrame(scaled, columns=X.columns)
