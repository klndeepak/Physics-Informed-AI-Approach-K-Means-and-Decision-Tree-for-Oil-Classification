# Decision Tree and K-Means Analysis of Raman Spectra for Edible Oils: A Physics-Informed AI Approach

## Overview

This repository contains the datasets, Jupyter notebooks, and generated figures used for Raman-spectroscopy-based analysis of edible oils in pure form and within fried-food (potato-chip) matrices.

The study combines unsupervised learning, interpretable machine learning, and Physics-Informed Artificial Intelligence (PI-AI) to investigate the intrinsic organization of Raman spectral data and its influence on classification performance, following a Mutually Exclusive, Collectively Exhaustive (MECE) analytical design: t-SNE and K-Means characterize intrinsic spectral organization, NNLS-based PI-AI isolates physically meaningful constituent contributions, and Decision Trees evaluate predictive separability and feature-level interpretability - each stage addressing a distinct, non-overlapping analytical question.

The workflow includes:

- K-Means clustering
- Elbow analysis
- Silhouette-score analysis
- t-SNE visualization
- Decision Tree classification
- Pre-pruned and post-pruned Decision Tree analysis
- Non-Negative Least Squares (NNLS)-based matrix subtraction
- Physics-informed spectral analysis

## How to Navigate This Repository

**Verifying a specific number or figure from the paper?** Every reported result traces to one notebook cell and one file in `Images/`:

| Paper content | Notebook | Key output files |
|---|---|---|
| t-SNE, elbow, silhouette (Figures 2-4, S1-S3, Table 1) | `K-Means Clusters_Oils.ipynb`, `K-Means Clusters_Chips.ipynb` | `Images/K-Means Clusters/tsne_*`, `elbow_plot_*`, `silhouette_*plot.jpg`, `kmeans_tsne_clusters_*.jpg` |
| MinMaxScaler robustness check | `K-Means Clusters_Oils - MinMaxScaler.ipynb` | (self-contained; not written to `Images/`) |
| Baseline / pre-pruned / post-pruned Decision Trees, all four datasets (Figures 5-11, Table 2) | `Decision Tree_Oils.ipynb`, `Decision Tree_Chips.ipynb`, `Decision Tree_Chips - Paper Subtracted.ipynb`, `Decision Tree_Chips - Paper and Potato Subtracted.ipynb` | `Images/Oils/`, `Images/Chips/`, `Images/Chips-Paper Subtracted/`, `Images/Chips-Paper and Potato Subtracted/` |
| Four-Raman-variable reproduction check (pure oils) | `Decision Tree_Oils.ipynb`, final section | `Images/Oils/FourFeature_Decision_Tree_*.jpg`, `confusion_matrix_test_4_wavenumbers_oils.jpg`, `df_features.csv` |
| Library/environment versions used for verification | any K-Means notebook | `Images/K-Means Clusters/requirements.md` |

**Want to re-run everything and confirm the numbers reproduce?** Use the Python package instead of re-executing notebooks by hand: see "Reproducible Python Pipeline" below, or just run `pytest` - the automated suite in `tests/` recomputes t-SNE coordinates, scaling behavior, and Decision Tree invariance properties, and diffs them against the checked-in results.

**Want to read the research narrative in full, as originally developed?** The seven root-level `.ipynb` files are the canonical, self-contained record - open them directly in Jupyter; each is heavily commented with the same reasoning that appears in the paper.

**Want to extend or adapt the analysis (e.g. a new dataset, a different classifier)?** Start in `src/raman_analysis/` - `data.py` and `datasets.py` for loading/column conventions, `clustering/` and `decision_tree/` for the two analysis families, and `tests/` for the regression suite any change should keep passing.

**Repository layout at a glance:**

- `*.csv` (root) - the four Raman spectral datasets (see "Datasets" below).
- `*.ipynb` (root) - the canonical notebooks (see "Reproducible Python Pipeline" below).
- `src/raman_analysis/` - the same analyses as an installable, tested Python package.
- `scripts/` - one command-line entry point per notebook workflow, plus `run_all.py`.
- `tests/` - the numerical regression suite (`pytest`).
- `Images/` - every figure, confusion matrix, feature-importance plot, and supporting table this project generates, organized one subfolder per dataset.
- `environment.yml` / `requirements.txt` / `pyproject.toml` - the pinned, verified computational environment.

## Datasets

Four Raman spectral datasets are provided:

1. `Oils.csv`
   - Raman spectra of the five pure edible oils.

2. `Chips.csv`
   - Raman spectra obtained from fried potato-chip samples containing the edible oils.

3. `Paper Subtracted.csv`
   - Chip spectra after removal of the paper contribution using NNLS-based spectral decomposition.

4. `Paper and Potato Subtracted.csv`
   - Chip spectra after accounting for paper and potato matrix contributions using NNLS-based spectral decomposition.

The NNLS approach incorporates the physical constraint that Raman spectral contributions are non-negative and additive (see "Physics-Informed AI Using NNLS" below for the full decomposition model).

## Normalization and Standardization

Two of the four datasets (`Oils.csv`, `Chips.csv`) arrive **column-wise (per-wavenumber) z-score normalized already** - each wavenumber's intensity is centered and scaled across the sample population before the CSV was ever written. This is verifiable from the data itself (every wavenumber column has population mean ≈ 0 and population std ≈ 1), but the pre-normalization raw intensities are not available in this repository, so the original upstream normalization cannot be independently re-derived here. The other two datasets (`Paper Subtracted.csv`, `Paper and Potato Subtracted.csv`) are NNLS difference-spectrum residuals and are **not** normalized.

Why column-wise (per-wavenumber), not row-wise (per-spectrum): each wavenumber is a distinct Raman-active vibrational mode with its own intrinsic scattering cross-section, and cross-sections differ by orders of magnitude between modes for physical reasons unrelated to how well a wavenumber discriminates between oils (e.g. C-H stretching, ~2850-2950 cm⁻¹, is characteristically far more Raman-intense than many fingerprint-region skeletal modes below 1500 cm⁻¹). Column-wise z-scoring ("autoscaling") puts every vibrational mode on equal statistical footing across the sample population - the standard chemometric remedy, and what makes the K-Means/t-SNE clustering results below meaningful. Row-wise normalization (e.g. Standard Normal Variate) addresses a different problem - multiplicative drift between acquisitions (laser power, exposure, sample positioning) - and is not attempted anywhere in this project.

**Clustering (K-Means / t-SNE):** runs only on the pure-oils and chips datasets, and always applies an explicit `StandardScaler` before any distance-based computation. Because those two datasets are already column-standardized, this call is a defensive, explicit re-application - it keeps the pipeline correct even if it is ever pointed at a dataset that is not pre-scaled, and it makes the normalization step visible in code rather than an invisible upstream fact.

**Decision Trees:** as of this version, all four dataset configurations apply an explicit, uniform column-wise z-score normalization (`sklearn.preprocessing.StandardScaler`, fit on the training split only, transformed onto the held-out test split) before any Decision Tree is fit. A Decision Tree's splits are threshold cuts on one feature at a time, and any per-feature affine rescaling - which z-scoring is - preserves each feature's sample rank order, so it cannot change which side of a split a sample falls on. **Accuracy, confusion matrices, and feature-importance ranking are therefore identical with or without this step** (verified directly in `tests/test_scaling_invariance.py`); only the numeric units of reported split thresholds and feature-importance axes change. It is applied uniformly so that the two previously-unnormalized NNLS-subtracted datasets are treated identically to the two pre-normalized ones, and so every reported wavenumber value across all four Decision Tree analyses is expressed in the same, comparable unit.

**Negative and zero values in plotted spectra are left as-is, not shifted.** The pure-oils mean-spectrum overview and the K-Means cluster-profile plots are drawn directly from the data's Z-scores, without shifting them to a positive floor. A negative or zero value there is a normal, statistically meaningful reading - this wavenumber's intensity at or below its own population mean - not an instrument artifact, so there is no principled floor to shift it to; their y-axes are labeled "Standardized Intensity (z-score)" to make that explicit.

## Reproducible Python Pipeline

The six workflow notebooks (below) remain the canonical research record. Their
analyses are also available as a modular Python package under
`src/raman_analysis/`, with one command-line script per notebook workflow.
Both implementations read the same four root-level CSV files and write to the
existing `Images/` hierarchy.

A seventh notebook, `K-Means Clusters_Oils - MinMaxScaler.ipynb`, is a
secondary robustness check (see "K-Means Clustering and t-SNE Analysis"
below for its result). It is not part of the modular `src/raman_analysis/`
package and has no corresponding script, since it exists to sanity-check
the main workflow rather than to run as part of it.

Create the verified environment and install the package:

```bash
conda env create -f environment.yml
conda activate raman-analysis
pip install -e .
```

Alternatively, create a virtual environment and run
`pip install -e .`. Development dependencies are available with
`pip install -e ".[dev]"`.

Run an individual analysis or the complete pipeline:

```bash
python scripts/run_kmeans_oils.py
python scripts/run_kmeans_chips.py
python scripts/run_kmeans_comparison.py

python scripts/run_decision_tree_oils.py
python scripts/run_decision_tree_chips.py
python scripts/run_decision_tree_chips_paper_subtracted.py
python scripts/run_decision_tree_chips_paper_and_potato_subtracted.py

python scripts/run_all.py
```

The package pins random seeds and preserves the notebooks' data preparation,
model parameters, metrics, confusion matrices, t-SNE coordinates, and output
filenames. Run `pytest` to verify the numerical regression suite.

## K-Means Clustering and t-SNE Analysis

K-Means clustering, elbow analysis, silhouette-score analysis, and t-SNE visualization were performed on the pure-oil and original chips datasets.

The corresponding notebooks are:

- `K-Means Clusters_Oils.ipynb`
- `K-Means Clusters_Chips.ipynb`

These analyses were used to investigate intrinsic spectral organization, cluster compactness, class overlap, and separability between pure oils and fried-food samples. At k=5 (matching the five known oil classes), the pure-oil spectra produced a silhouette score of 0.343, versus 0.126 for the fried-food spectra - quantitative confirmation that pure oils carry substantially stronger intrinsic class structure than the same oils inside the chip matrix.

`K-Means Clusters_Oils - MinMaxScaler.ipynb` repeats the pure-oils clustering analysis with `MinMaxScaler` in place of `StandardScaler`, as a robustness check on that preprocessing choice (see "Normalization and Standardization" above). At k=5, `MinMaxScaler` produced a lower silhouette score than `StandardScaler` (0.325 vs. 0.343), consistent with `StandardScaler` better preserving relative variation across features than a scaler whose range is set by two extreme data points.

## Decision Tree Analysis

Decision Tree analysis was performed for four datasets:

- Pure oils
- Original chips
- Paper-subtracted chips
- Paper- and potato-subtracted chips

The corresponding notebooks are:

- `Decision Tree_Oils.ipynb`
- `Decision Tree_Chips.ipynb`
- `Decision Tree_Chips - Paper Subtracted.ipynb`
- `Decision Tree_Chips - Paper and Potato Subtracted.ipynb`

The Decision Tree workflow includes baseline, pre-pruned, and post-pruned models and evaluates classification performance using metrics such as accuracy, precision, recall, and F1-score. Test-set accuracy for each dataset and stage:

| Dataset | Baseline | Pre-pruned | Post-pruned |
|---|---|---|---|
| Pure oils | 100% | 100% | 100% |
| Original chips | 62.6% | 64.8% | 64.4% |
| Paper-subtracted chips | 77.0% | 73.7% | 79.3% |
| Paper-and-potato-subtracted chips | 73.7% | 76.7% | 79.6% |

NNLS-based matrix subtraction (below) is the main driver of improvement for the chips datasets - pruning alone, without it, only modestly improves the original chips' generalization.

## Physics-Informed AI Using NNLS

Each fried-chip Raman spectrum is modeled as a non-negative, additive combination of three reference components - oil, paper, and potato:

```
X_chips = C_oil * X_oil + C_potato * X_potato + C_paper * X_paper + epsilon,   subject to C_oil, C_potato, C_paper, epsilon >= 0
```

The non-negativity constraint reflects the physical reality that Raman spectral contributions cannot be negative; the coefficients are estimated by constrained least-squares (NNLS). Subtracting the fitted paper term alone gives `Paper Subtracted.csv`; subtracting both fitted paper and potato terms gives `Paper and Potato Subtracted.csv`.

The matrix-corrected datasets were then used for Decision Tree classification to determine whether removal of non-oil spectral contributions could improve class separability and model performance. NNLS-based preprocessing improved test-set classification performance for both matrix-containing chips datasets (see the table above) while substantially reducing model complexity - the post-pruned model's count of non-zero-importance wavenumbers fell from 29 (original chips) to 5 (paper-subtracted) and 4 (paper-and-potato-subtracted).

Two Raman bands - ~1652 cm⁻¹ (C=C stretching, lipid unsaturation) and ~1127 cm⁻¹ (C-C skeletal stretching, hydrocarbon-chain organization) - persist as important features across every dataset, and their combined share of total feature importance grows as matrix interference is removed: **50%** in pure oils, **~62%** after paper subtraction, and **~89%** after paper-and-potato subtraction. This indicates that these two bands are the most robust, matrix-independent oil-specific Raman signatures identified in this study.

## Images Directory Reference

Every figure, confusion matrix, feature-importance plot, and supporting table this project generates is organized under `Images/`, one subfolder per dataset (see "How to Navigate This Repository" above for the top-level repository layout):

```text
Images/
│
├── Chips/
│   ├── Decision Tree plots
│   ├── Confusion matrices
│   ├── Feature-importance plots
│   ├── Pre-pruned Decision Tree
│   └── Post-pruned Decision Tree
│
├── Chips-Paper Subtracted/
│   ├── Decision Tree plots
│   ├── Confusion matrices
│   ├── Feature-importance plots
│   └── Pruned Decision Tree results
│
├── Chips-Paper and Potato Subtracted/
│   ├── Decision Tree plots
│   ├── Confusion matrices
│   ├── Feature-importance plots
│   └── Pruned Decision Tree results
│
├── K-Means Clusters/
│   ├── Elbow plots
│   ├── Silhouette plots
│   ├── K-Means cluster visualizations
│   ├── t-SNE plots
│   ├── 2D and 3D t-SNE visualizations
│   ├── Cluster profiles
│   └── Supporting t-SNE data
│
└── Oils/
    ├── Raman spectra
    ├── Decision Tree plots
    ├── Confusion matrices
    ├── Feature-importance plots
    ├── Pre-pruned Decision Tree
    └── Post-pruned Decision Tree

```

## Analysis Workflow

```text
Raman Spectral Data
        │
        ├── Pure Oils ────────────────┐
        │                             │
        └── Fried-Food (Chips) ──────┤
                                      │
                                      ▼
                              Spectral Analysis
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                         ▼                         ▼
                  K-Means / t-SNE          Decision Tree
                         │                         │
                         │                  ┌──────┴──────┐
                         │                  │             │
                         │               Baseline     Pruning
                         │                              │
                         │                       Pre-pruned /
                         │                       Post-pruned
                         │
                         ▼
                  Spectral Organization
                  and Class Separability
                                                   
Fried-Food Spectra
        │
        ▼
 NNLS Spectral Decomposition
        │
        ├── Paper Subtraction
        │
        └── Paper + Potato Subtraction
        │
        ▼
 Matrix-Corrected Raman Spectra
        │
        ▼
 Decision Tree Classification

```

## Key Findings

- **Pure oils show stronger intrinsic class organization than fried-food samples**, both qualitatively (t-SNE) and quantitatively (silhouette score 0.343 vs. 0.126 at k=5).
- **Perfect, minimal-feature classification of pure oils**: the pre-pruned and post-pruned Decision Trees independently select the *same* four Raman variables (~1127, ~1273, ~1322, ~1649 cm⁻¹) - only 0.21% of the original 1866-variable spectral space - and both achieve 100% train and test accuracy with them.
- **That four-feature representation cuts the in-memory data footprint by 99.44%** (14.3 MB -> 81.8 KB), with no loss of classification accuracy.
- **NNLS-based matrix subtraction is the main driver of improved chips classification and reduced model complexity**: post-pruned test accuracy rises from 64.4% (original chips) to 79.3% (paper-subtracted) and 79.6% (paper-and-potato-subtracted), while the number of important wavenumbers falls from 29 to 5 and 4, respectively.
- **Two Raman bands - lipid unsaturation (~1652 cm⁻¹) and hydrocarbon-chain skeletal organization (~1127 cm⁻¹) - are the most robust, matrix-independent oil-specific signatures**: their combined feature importance rises from 50% in pure oils to ~62% and ~89% as paper and potato contributions are progressively removed.

Collectively, these results demonstrate that accurate, physically interpretable Raman-based oil identification is achievable through highly compact spectral representations - a promising foundation for Frugal AI, Edge AI, and embedded food-quality monitoring.

## Evaluation Methodology

Every Decision Tree stage (baseline, pre-pruned, post-pruned) reports accuracy, precision, recall, F1, and a full confusion matrix for **both** the training set and an independent, held-out test set. Five-fold cross-validation is used strictly for hyperparameter selection on the training split, never as a substitute for test-set evaluation - so every headline number in this project reflects genuine held-out performance, not an in-sample or cross-validation-only figure.

This is a 5-class (oil-type) classification problem, not a binary detection task, so metrics are computed with scikit-learn's weighted multi-class averaging (`average="weighted"`) across all five classes - the direct multi-class generalization of the sensitivity/precision/recall reported for a binary problem's single positive class. The complete per-class confusion matrix, generated for every stage and dataset, additionally makes any specific oil-pair confusion pattern (e.g. how often one oil is mistaken for another) directly readable, beyond the aggregate scalar metrics.

## Scope and Future Work

This study is built on a controlled, balanced experimental cohort: 50 independent pure-oil preparations (5 oil types × 10 replicates; see `Oils.csv`'s `Oil_Type` column) and 45 independent fried-chip batches (5 oil types × 9 frying cycles; see `Chips.csv`'s `Chips Type` column), each characterized by ~20 Raman spectra collected at different spatial points - a standard approach in Raman spectroscopy for obtaining a robust, averaged spectral signature per physical sample.

All classification and clustering results in this study are evaluated at the spectrum level across this cohort. As the paper itself notes (§3.5.1), identical train/test splits are maintained across every baseline, pre-pruned, and post-pruned comparison, which is precisely what makes the reported improvements - from NNLS-based matrix correction and from pruning - directly attributable to the change being tested rather than to split variation. This spectrum-level evaluation is well matched to the study's central objective: identifying a minimal, physically interpretable set of discriminative Raman variables and quantifying how matrix correction reshapes them.

A valuable next step, complementary to the future directions already identified for this work (additional food matrices, adulterated and recycled frying oils, varied frying/storage conditions), is to validate the identified spectral markers under a batch-independent (sample-level) split as the physical sample cohort grows in subsequent studies - directly testing generalization to entirely new preparation batches. `sklearn.model_selection.GroupShuffleSplit`/`GroupKFold`, keyed on the `Oil_Type`/`Chips Type` identifiers already present in the data, provide a ready-made path for that extension without requiring new data collection to get started.

## Reproducibility

All datasets, Jupyter notebooks, and generated figures required to reproduce the analyses presented in the study are provided in this repository.

The notebooks can be executed in Jupyter Notebook or JupyterLab after installing the required Python packages.

The production scripts provide a non-interactive replacement for the notebook
execution path. Dependency versions used for numerical verification are pinned
in `environment.yml` and `requirements.txt`; automated checks live in `tests/`.

## Citation

If you use this repository or the associated analysis, please cite:

> Shaw, A., Chandrasekar, S. N., Muthukumar, S. V., Gupta, J., Kallepalli, D. L. N. (2026). Decision Tree and K-Means Analysis of Raman Spectra for Edible Oils: A Physics-Informed AI Approach. arXiv:2608.20440. https://arxiv.org/abs/2608.20440

```bibtex
@misc{shaw2026decisiontree,
  title         = {Decision Tree and K-Means Analysis of Raman Spectra for Edible Oils: A Physics-Informed AI Approach},
  author        = {Shaw, Amrita and Chandrasekar, S. N. and Muthukumar, Sai V. and Gupta, Jhinuk and Kallepalli, Deepak L. N.},
  year          = {2026},
  eprint        = {2608.20440},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2608.20440}
}
```

## License

All rights reserved - see [`LICENSE`](LICENSE). No part of this repository (code, notebooks, datasets, or generated figures) may be used, copied, modified, or redistributed without prior express written permission. This matches the Data Availability statement in the associated publication: access is granted upon reasonable request, on a case-by-case basis. To request permission, contact the corresponding author(s) of the associated publication.

## Authors

Amrita Shaw, Chandrasekar S. N., Sai Muthukumar V., Jhinuk Gupta, Deepak L. N. Kallepalli
