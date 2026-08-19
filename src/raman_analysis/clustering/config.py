"""Per-dataset configuration for the K-Means / t-SNE clustering pipeline.

Both datasets share the same output directory, ``Images/K-Means
Clusters/``, and the same shape of workflow (scale -> t-SNE -> elbow ->
silhouette -> K-Means -> cluster profile); what differs is captured
here: dataset identity (see ``datasets.py``), the color palette and
legend order for each oil label style ("GNO" for pure oils vs "GNO C"
for chips), the perplexity sweep, and output filenames.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import (
    CHIPS_OIL_PALETTE,
    CHIPS_OIL_PALETTE_FLAT,
    OIL_CODES,
    OIL_PALETTE,
    OIL_PALETTE_FLAT,
)
from ..datasets import CHIPS_IDENTITY, OILS_IDENTITY, SpectralDatasetIdentity
from ..paths import IMAGES_DIR

CLUSTER_OUTPUT_DIR = IMAGES_DIR / "K-Means Clusters"

# Number of oil classes; also the K used for the final K-Means model,
# matching the original notebooks' choice to cluster into exactly as
# many groups as there are known oil types.
N_OIL_CLASSES = 5


@dataclass(frozen=True)
class ClusteringDatasetConfig:
    key: str
    identity: SpectralDatasetIdentity
    hue_order: list[str]
    palette: dict[str, str]
    palette_flat: dict[str, str]
    perplexities: list[int]
    filenames: dict[str, str]
    # Column name used for the oil-label column added to every t-SNE
    # DataFrame (2D/3D, perplexity grid). Both datasets store the same
    # label *values*; only the column name differs, matching the two
    # original notebooks exactly - this shows up in one place a reader
    # can actually see it: the auto-generated legend title on the first
    # panel of the perplexity grid (every other plot sets an explicit
    # legend title regardless of this name).
    hue_column_name: str
    has_average_vs_random_plot: bool = False
    tsne_default_legend_loc: str | None = None

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


OILS = ClusteringDatasetConfig(
    key="oils",
    identity=OILS_IDENTITY,
    hue_order=OIL_CODES,
    palette=OIL_PALETTE,
    palette_flat=OIL_PALETTE_FLAT,
    perplexities=[5, 10, 20, 30, 50, 75],
    hue_column_name="Oil_Type",
    has_average_vs_random_plot=True,
    tsne_default_legend_loc="upper left",
    filenames={
        "tsne_default": "tsne_plot_default_oils.jpg",
        "tsne_perplexity_grid": "tsne_perplexity_grid_oils.jpg",
        "tsne_2d_csv": "tsne_2d_data_with_oils.csv",
        "tsne_3d_csv": "tsne_3d_with_oil_labels.csv",
        "tsne_3d_pairwise": "tsne_3d_pairwise_projections_Oils.png",
        "tsne_3d_plotly": "tsne_3d_oils_plot.jpg",
        "elbow_plot": "elbow_plot_oils.jpg",
        "silhouette_plot": "silhouette_oils_plot.jpg",
        "kmeans_tsne_clusters": "kmeans_tsne_clusters_oils.jpg",
        "cluster_profiles": "cluster_profiles_broken_axis.jpg",
        "cluster_average_vs_random": "cluster_average_vs_random_broken_axis.jpg",
    },
)

CHIPS = ClusteringDatasetConfig(
    key="chips",
    identity=CHIPS_IDENTITY,
    hue_order=[f"{code} C" for code in OIL_CODES],
    palette=CHIPS_OIL_PALETTE,
    palette_flat=CHIPS_OIL_PALETTE_FLAT,
    perplexities=[5, 10, 20, 40, 50, 75],
    hue_column_name="Oil Type",
    filenames={
        "tsne_default": "tsne_plot_default_chips.jpg",
        "tsne_perplexity_grid": "tsne_perplexity_grid_chips.jpg",
        "tsne_2d_csv": "tsne_2d_data_with_chips.csv",
        "tsne_3d_csv": "tsne_3d_with_chips_labels.csv",
        "tsne_3d_pairwise": "tsne_3d_pairwise_projections_Chips.png",
        "tsne_3d_plotly": "tsne_3d_chips_plot.jpg",
        "elbow_plot": "elbow_plot_chips.jpg",
        "silhouette_plot": "silhouette_chips_plot.jpg",
        "kmeans_tsne_clusters": "kmeans_tsne_clusters_chips.jpg",
        "cluster_profiles": "cluster_profiles_broken_axis_chips.jpg",
    },
)

DATASETS = {config.key: config for config in (OILS, CHIPS)}
