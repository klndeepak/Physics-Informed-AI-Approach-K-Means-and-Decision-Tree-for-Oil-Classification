"""Z-score scaling, the required first step before any distance-based
clustering (K-Means, t-SNE) on spectra whose wavenumber columns can span
very different intensity ranges.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """Z-score scale every column of ``X`` and return it as a DataFrame."""
    scaled = StandardScaler().fit_transform(X)
    return pd.DataFrame(scaled, columns=X.columns)
