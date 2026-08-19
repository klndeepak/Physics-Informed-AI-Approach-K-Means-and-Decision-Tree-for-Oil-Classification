"""t-SNE embeddings and every plot built from them.

t-SNE is used purely for visualization here (K-Means itself clusters on
the full, scaled spectrum - see ``kmeans_clusters.py``): a 2D view at
the default perplexity, a small-multiples grid sweeping perplexity to
check how stable the neighborhood structure is, and a fixed
perplexity=50 2D/3D pair used for every downstream cluster-vs-true-label
comparison plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.manifold import TSNE

DEFAULT_PERPLEXITY = 30.0
FIXED_PERPLEXITY = 50
LEGEND_TITLE = "Oil Type"


def run_tsne(
    scaled_df: pd.DataFrame, n_components: int, perplexity: float, random_state: int
) -> pd.DataFrame:
    """Fit t-SNE and return the embedding as a "Feature 1", "Feature 2", ... DataFrame."""
    tsne = TSNE(
        n_components=n_components, perplexity=perplexity, n_jobs=-2, random_state=random_state
    )
    embedding = tsne.fit_transform(scaled_df)
    columns = [f"Feature {i + 1}" for i in range(n_components)]
    return pd.DataFrame(embedding, columns=columns)


def plot_tsne_default(
    tsne_2d: pd.DataFrame,
    y: pd.Series,
    hue_column_name: str,
    palette: dict[str, str],
    legend_loc: str | None,
    out_path: Path,
    dpi: int = 300,
) -> pd.DataFrame:
    """Scatter the fixed-perplexity 2D embedding, colored by true label."""
    tsne_2d = tsne_2d.copy()
    tsne_2d[hue_column_name] = y.reset_index(drop=True)

    sns.scatterplot(
        data=tsne_2d, x="Feature 1", y="Feature 2", hue=hue_column_name, palette=palette
    )
    legend_kwargs = {"title": LEGEND_TITLE, "bbox_to_anchor": (1.05, 1)}
    if legend_loc is not None:
        legend_kwargs["loc"] = legend_loc
    plt.legend(**legend_kwargs)
    plt.title("t-SNE 2D Visualization (Colored by Oil Type - True Labels)")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return tsne_2d


def plot_perplexity_grid(
    scaled_df: pd.DataFrame,
    y: pd.Series,
    hue_column_name: str,
    hue_order: list[str],
    palette: dict[str, str],
    perplexities: list[int],
    random_state: int,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """3x2 grid of 2D t-SNE embeddings, one per perplexity in ``perplexities``."""
    plt.figure(figsize=(20, 15))
    for i, perplexity in enumerate(perplexities):
        embedding = run_tsne(
            scaled_df, n_components=2, perplexity=perplexity, random_state=random_state
        )
        embedding[hue_column_name] = y.reset_index(drop=True)

        plt.subplot(3, 2, i + 1)
        sns.scatterplot(
            data=embedding, x="Feature 1", y="Feature 2",
            hue=hue_column_name, hue_order=hue_order, palette=palette,
        )
        plt.title(f"Perplexity = {perplexity}")
        if i != 0:
            plt.legend([], [], frameon=False)

    plt.tight_layout(pad=2)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_and_verify_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Save ``df`` to CSV, then read it back and print a quick summary.

    The read-back is a sanity check carried over from the notebooks: it
    confirms the file was written correctly before later steps rely on
    it existing on disk.
    """
    df.to_csv(out_path, index=False)
    written = pd.read_csv(out_path)
    print(written.columns)
    print(written.shape)
    print(written.nunique())
    print(written.head())


def print_feature_ranges(
    df: pd.DataFrame, hue_column_name: str, feature_columns: list[str]
) -> None:
    aggregation = {column: ["min", "max"] for column in feature_columns}
    print(df.groupby(hue_column_name).agg(aggregation))


def plot_pairwise_projections(
    tsne_3d: pd.DataFrame,
    hue_column_name: str,
    palette: dict[str, str],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Feature1-2, Feature2-3, and Feature3-1 2D projections of the 3D embedding."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pairs = [("Feature 1", "Feature 2"), ("Feature 2", "Feature 3"), ("Feature 3", "Feature 1")]

    for ax, (x_col, y_col) in zip(axes, pairs):
        sns.scatterplot(
            data=tsne_3d, x=x_col, y=y_col, hue=hue_column_name,
            palette=palette, alpha=0.7, s=60, ax=ax,
        )
        ax.set_title(f"{x_col} vs {y_col}")
        ax.grid(alpha=0.3)

    axes[1].legend_.remove()
    axes[2].legend_.remove()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_3d_plotly(
    tsne_3d: pd.DataFrame, y: pd.Series, palette_flat: dict[str, str], out_path: Path
) -> None:
    """Interactive 3D scatter of the fixed-perplexity embedding, saved as a static image."""
    tsne_3d = tsne_3d.copy()
    tsne_3d["True_Label"] = y.values

    fig = px.scatter_3d(
        tsne_3d, x="Feature 1", y="Feature 2", z="Feature 3",
        color="True_Label", color_discrete_map=palette_flat, title="t-SNE 3D (True Labels)",
    )
    fig.write_image(out_path)
