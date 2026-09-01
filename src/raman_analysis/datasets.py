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


# "Oil_Type" is not a throwaway metadata column: it has exactly 50 unique
# values (5 oil types x 10 physical replicate preparations) and is the
# physical-sample identifier behind Oils.csv's 1000 rows (20 spectra per
# physical sample). It is dropped here like any other metadata column,
# consistent with this study's spectrum-level evaluation scope - see the
# "SCOPE" note in decision_tree/pipeline.py, and the README's "Scope and
# Future Work" section for the sample-level-validation extension this
# identifier is ready to support.
OILS_IDENTITY = SpectralDatasetIdentity(
    csv_path_name="Oils.csv",
    drop_unnamed_index=True,
    drop_columns=["Oil_Type", "Oil"],
    target_column="Oil",
)

# "Chips Type" is similarly the physical-sample identifier, not just
# metadata: exactly 45 unique values (5 oil types x 9 frying cycles)
# behind Chips.csv's 900 rows (20 spectra per physical batch). Same note
# as "Oil_Type" above.
CHIPS_IDENTITY = SpectralDatasetIdentity(
    csv_path_name="Chips.csv",
    drop_unnamed_index=True,
    drop_columns=["Oil Type", "Chips Type"],
    target_column="Oil Type",
)
