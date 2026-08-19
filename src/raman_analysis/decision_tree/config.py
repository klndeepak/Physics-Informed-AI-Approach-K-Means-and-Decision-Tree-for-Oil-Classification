"""Per-dataset configuration for the Decision Tree pipeline.

Four spectral datasets share one pipeline (see ``pipeline.py``): the
pure oils, the raw fried-chip spectra, and two NNLS matrix-corrected
chip variants (paper subtracted, paper-and-potato subtracted). What
differs between them is data plumbing (which metadata columns to drop,
what the target column is called) and where each figure gets written -
all of which is captured here as one :class:`DecisionTreeDatasetConfig`
per dataset, built once in ``DATASETS`` below.

Output filenames intentionally keep the exact (and occasionally
inconsistent-looking, e.g. mixed case) names the original notebooks
used, so the regenerated files land on top of what is already committed
under ``Images/`` and stay directly diffable against it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..datasets import CHIPS_IDENTITY, OILS_IDENTITY, SpectralDatasetIdentity
from ..paths import IMAGES_DIR

# Spectral rows/columns are already de-meaned to zero on this dataset, so
# the notebook only *reports* the minimum instead of shifting it, and
# frames the message accordingly.
_ZSCORE_NOTE = "already Z-score normalized"
_DIFFERENCE_SPECTRUM_NOTE = "a difference spectrum"

# The two NNLS matrix-corrected chip variants are Decision-Tree-only
# (the clustering pipeline never analyzes them), so their identity is
# defined here rather than alongside the shared Oils/Chips identities in
# ``datasets.py``.
_PAPER_SUBTRACTED_IDENTITY = SpectralDatasetIdentity(
    csv_path_name="Paper Subtracted.csv",
    drop_unnamed_index=False,
    drop_columns=["Oil", "Stage", "Replicate", "Chips Type"],
    target_column="Oil",
)
_PAPER_AND_POTATO_SUBTRACTED_IDENTITY = SpectralDatasetIdentity(
    csv_path_name="Paper and Potato Subtracted.csv",
    drop_unnamed_index=False,
    drop_columns=["Oil", "Stage", "Replicate", "Chips Type"],
    target_column="Oil",
)


@dataclass(frozen=True)
class DecisionTreeDatasetConfig:
    key: str
    identity: SpectralDatasetIdentity
    output_dir_name: str
    negative_value_note: str
    filenames: dict[str, str]
    has_mean_spectra_plot: bool = False
    has_four_feature_check: bool = False

    @property
    def csv_path(self):
        return self.identity.csv_path

    @property
    def drop_unnamed_index(self) -> bool:
        return self.identity.drop_unnamed_index

    @property
    def drop_columns(self) -> list[str]:
        return self.identity.drop_columns

    @property
    def target_column(self) -> str:
        return self.identity.target_column

    @property
    def output_dir(self):
        return IMAGES_DIR / self.output_dir_name


OILS = DecisionTreeDatasetConfig(
    key="oils",
    identity=OILS_IDENTITY,
    output_dir_name="Oils",
    negative_value_note=_ZSCORE_NOTE,
    has_mean_spectra_plot=True,
    has_four_feature_check=True,
    filenames={
        "mean_spectra": "Raman_spectra_oils_random.jpg",
        "default_train_confusion": "default_confusion_matrix_train_oil.jpg",
        "default_test_confusion": "default_confusion_matrix_test_oil.jpg",
        "prepruned_train_confusion": "confusion_matrix_train_prepruned_oil.jpg",
        "prepruned_test_confusion": "confusion_matrix_test_prepruned_oil.jpg",
        "prepruned_tree_plot": "Prepruned_Decision_Tree_oils.jpg",
        "prepruned_text_report": "Prepruned_Decision_Tree_Text_Report_Oils.jpg",
        "prepruned_feature_importance": "decision_tree_feature_importance_oils.jpg",
        "important_features_csv": "df_features.csv",
        "postpruned_train_confusion": "confusion_matrix_train_postpruned_oil.jpg",
        "postpruned_test_confusion": "confusion_matrix_test_postpruned_oil.jpg",
        "postpruned_tree_plot": "postpruned_decision_tree_oils.jpg",
        "postpruned_text_report": "postpruned_tree_rules_oils.jpg",
        "postpruned_feature_importance": "PostPruned_Feature_Importance_oils.jpg",
    },
)

CHIPS = DecisionTreeDatasetConfig(
    key="chips",
    identity=CHIPS_IDENTITY,
    output_dir_name="Chips",
    negative_value_note=_ZSCORE_NOTE,
    filenames={
        "default_train_confusion": "confusion_matrix_train_Chips.jpg",
        "default_test_confusion": "confusion_matrix_Default_test_chips.jpg",
        "prepruned_train_confusion": "confusion_matrix_train_prepruned_chips.jpg",
        "prepruned_test_confusion": "confusion_matrix_test_prepruned_chips.jpg",
        "prepruned_tree_plot": "Prepruned_Decision_Tree_chips.jpg",
        "prepruned_text_report": "Prepruned_Decision_Tree_Text_Report_Chips.jpg",
        "prepruned_feature_importance": "decision_tree_feature_importance_Chips.jpg",
        "postpruned_train_confusion": "confusion_matrix_train_postpruned_Chips.jpg",
        "postpruned_test_confusion": "confusion_matrix_test_postpruned_chips.jpg",
        "postpruned_tree_plot": "postpruned_decision_tree_Chips.jpg",
        "postpruned_text_report": "postpruned_tree_rules_Chips.jpg",
        "postpruned_feature_importance": "PostPruned_Feature_Importance_Chips.jpg",
    },
)

CHIPS_PAPER_SUBTRACTED = DecisionTreeDatasetConfig(
    key="chips_paper_subtracted",
    identity=_PAPER_SUBTRACTED_IDENTITY,
    output_dir_name="Chips-Paper Subtracted",
    negative_value_note=_DIFFERENCE_SPECTRUM_NOTE,
    filenames={
        "default_train_confusion": "default_confusion_matrix_train_Chips_Paper.jpg",
        "default_test_confusion": "default_confusion_matrix_test_chips_paper.jpg",
        "prepruned_train_confusion": "confusion_matrix_train_prepruned_Chips_Paper.jpg",
        "prepruned_test_confusion": "confusion_matrix_test_prepruned_Chips_Paper.jpg",
        "prepruned_tree_plot": "Prepruned_Decision_Tree_Chips_Paper.jpg",
        "prepruned_text_report": "Prepruned_Decision_Tree_Text_Report_Chips_Paper.jpg",
        "prepruned_feature_importance": "decision_tree_feature_importance_Chips_Paper.jpg",
        "postpruned_train_confusion": "Confusion_matrix_train_postpruned_Chips_Paper.jpg",
        "postpruned_test_confusion": "confusion_matrix_test_postpruned_Chips_Paper.jpg",
        "postpruned_tree_plot": "postpruned_decision_tree_Chips_Paper.jpg",
        "postpruned_text_report": "postpruned_tree_rules_Chips_Paper.jpg",
        "postpruned_feature_importance": "PostPruned_Feature_Importance_Chips_Paper.jpg",
    },
)

CHIPS_PAPER_AND_POTATO_SUBTRACTED = DecisionTreeDatasetConfig(
    key="chips_paper_and_potato_subtracted",
    identity=_PAPER_AND_POTATO_SUBTRACTED_IDENTITY,
    output_dir_name="Chips-Paper and Potato Subtracted",
    negative_value_note=_DIFFERENCE_SPECTRUM_NOTE,
    filenames={
        "default_train_confusion": "default_confusion_matrix_train_Chips_Paper_Potato.jpg",
        "default_test_confusion": "default_confusion_matrix_test_Chips_Paper_Potato.jpg",
        "prepruned_train_confusion": "confusion_matrix_train_prepruned_Chips_Paper_Potato.jpg",
        "prepruned_test_confusion": "confusion_matrix_test_prepruned_Chips_Paper_Potato.jpg",
        "prepruned_tree_plot": "Prepruned_Decision_Tree_chips-paper and potato.jpg",
        "prepruned_text_report": "Prepruned_Decision_Tree_Text_Report_chips-paper and potato.jpg",
        "prepruned_feature_importance": (
            "decision_tree_feature_importance_chips-paper and potato.jpg"
        ),
        "postpruned_train_confusion": "confusion_matrix_train_postpruned_Chips_Paper_Potato.jpg",
        "postpruned_test_confusion": "confusion_matrix_test_postpruned_Chips_Paper_Potato.jpg",
        "postpruned_tree_plot": "postpruned_decision_tree_Chips-Paper and Potato.jpg",
        "postpruned_text_report": "postpruned_tree_rules_Chips-Paper and Potato.jpg",
        "postpruned_feature_importance": (
            "PostPruned_Feature_Importance_Chips-Paper and Potato.jpg"
        ),
    },
)

DATASETS = {
    config.key: config
    for config in (OILS, CHIPS, CHIPS_PAPER_SUBTRACTED, CHIPS_PAPER_AND_POTATO_SUBTRACTED)
}
