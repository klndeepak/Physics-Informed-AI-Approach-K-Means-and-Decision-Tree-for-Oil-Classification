# Decision Tree and K-Means Analysis of Raman Spectra for Edible Oils: A Physics-Informed AI Approach

## Overview

This repository contains the datasets, Jupyter notebooks, and generated figures used for Raman-spectroscopy-based analysis of edible oils in pure form and within fried-food (potato-chip) matrices.

The study combines unsupervised learning, interpretable machine learning, and Physics-Informed Artificial Intelligence (PI-AI) to investigate the intrinsic organization of Raman spectral data and its influence on classification performance.

The workflow includes:

- K-Means clustering
- Elbow analysis
- Silhouette-score analysis
- t-SNE visualization
- Decision Tree classification
- Pre-pruned and post-pruned Decision Tree analysis
- Non-Negative Least Squares (NNLS)-based matrix subtraction
- Physics-informed spectral analysis

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

The NNLS approach incorporates the physical constraint that Raman spectral contributions are non-negative and additive.

## Reproducible Python Pipeline

The six notebooks remain the canonical research record. Their analyses are
also available as a modular Python package under `src/raman_analysis/`, with
one command-line script per notebook workflow. Both implementations read the
same four root-level CSV files and write to the existing `Images/` hierarchy.

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

These analyses were used to investigate intrinsic spectral organization, cluster compactness, class overlap, and separability between pure oils and fried-food samples.

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

The Decision Tree workflow includes baseline, pre-pruned, and post-pruned models and evaluates classification performance using metrics such as accuracy, precision, recall, and F1-score.

## Physics-Informed AI Using NNLS

NNLS-based spectral decomposition was used to reduce spectral contributions arising from the food matrix.

The matrix-corrected datasets were then used for Decision Tree classification to determine whether removal of non-oil spectral contributions could improve class separability and model performance.

The study showed that NNLS-based preprocessing improved classification performance for the matrix-containing chips datasets while also reducing model complexity.

## Repository Structure

```text
Decision Tree/
│
├── README.md
│
├── Oils.csv
├── Chips.csv
├── Paper Subtracted.csv
├── Paper and Potato Subtracted.csv
│
├── K-Means Clusters_Chips.ipynb
├── K-Means Clusters_Oils.ipynb
│
├── Decision Tree_Chips.ipynb
├── Decision Tree_Oils.ipynb
├── Decision Tree_Chips - Paper Subtracted.ipynb
├── Decision Tree_Chips - Paper and Potato Subtracted.ipynb
│
├── Images/
│   │
│   ├── Chips/
│   │   └── Decision Tree results for original chips spectra
│   │
│   ├── Chips-Paper Subtracted/
│   │   └── Decision Tree results for paper-subtracted spectra
│   │
│   ├── Chips-Paper and Potato Subtracted/
│   │   └── Decision Tree results for paper- and potato-subtracted spectra
│   │
│   ├── K-Means Clusters/
│   │   └── K-Means, t-SNE, elbow, and silhouette results
│   │
│   └── Oils/
│       └── Decision Tree results for pure-oil spectra
│
└── .ipynb_checkpoints/
    └── Jupyter Notebook checkpoint files


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
        └── Fried-­Food (Chips) ──────┤
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
                                                   
Fried-­Food Spectra
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

The pure-oil Raman spectra showed stronger intrinsic class organization and greater spectral separability than the fried-food samples.

The optimized Decision Tree models achieved perfect classification of the five pure edible oils using only four Raman variables from the original 1866-variable spectral space.

NNLS-based matrix correction improved classification performance for the chips datasets and reduced the number of important spectral variables required by the Decision Tree models.

These results demonstrate the potential of combining Raman spectroscopy, unsupervised learning, interpretable Decision Trees, and Physics-Informed AI for computationally efficient analysis of complex food matrices.

## Reproducibility

All datasets, Jupyter notebooks, and generated figures required to reproduce the analyses presented in the study are provided in this repository.

The notebooks can be executed in Jupyter Notebook or JupyterLab after installing the required Python packages.

The production scripts provide a non-interactive replacement for the notebook
execution path. Dependency versions used for numerical verification are pinned
in `environment.yml` and `requirements.txt`; automated checks live in `tests/`.

## Citation

If you use this repository or the associated analysis, please cite the corresponding research publication.

## Authors

Amrita Shaw, Chandrasekar S. N., Sai Muthukumar V., Jhinuk Gupta, Deepak L. N. Kallepalli


