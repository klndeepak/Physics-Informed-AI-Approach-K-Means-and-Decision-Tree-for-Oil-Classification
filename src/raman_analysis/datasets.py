"""Identity of the two datasets shared by both pipelines: what CSV to
read, which of its columns are metadata (not spectral), and which
metadata column is the classification target.

The Decision Tree pipeline additionally analyzes two NNLS-corrected chip
variants that the clustering pipeline never touches, so those two live
only in ``decision_tree/config.py``. Oils and Chips, however, are used
identically by both pipelines - defining their identity once here (
rather than in both ``decision_tree/config.py`` and
``clustering/config.py``) means a change to, say, which columns count as
metadata only has to happen in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from .paths import DATA_DIR


@dataclass(frozen=True)
class SpectralDatasetIdentity:
    csv_path_name: str
    drop_unnamed_index: bool
    drop_columns: list[str]
    target_column: str

    @property
    def csv_path(self):
        return DATA_DIR / self.csv_path_name


OILS_IDENTITY = SpectralDatasetIdentity(
    csv_path_name="Oils.csv",
    drop_unnamed_index=True,
    drop_columns=["Oil_Type", "Oil"],
    target_column="Oil",
)

CHIPS_IDENTITY = SpectralDatasetIdentity(
    csv_path_name="Chips.csv",
    drop_unnamed_index=True,
    drop_columns=["Oil Type", "Chips Type"],
    target_column="Oil Type",
)
