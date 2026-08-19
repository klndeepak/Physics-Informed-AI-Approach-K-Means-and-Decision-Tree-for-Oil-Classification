# Supplementary Results

This directory contains additional numerical checks derived from the same four
root-level datasets used by the core analysis pipeline.

- `tables/` contains the exact values in CSV format.
- `figures/` contains PNG summaries generated from those CSV tables.

Regenerate every file with:

```bash
python scripts/run_supplementary.py
```

To redraw only the PNG files from existing CSV tables:

```bash
python scripts/run_supplementary.py --figures-only
```

Robustness tables use ten fixed train/test split seeds. Model hyperparameters
are selected on the canonical split and then held constant across those splits.
K-Means stability likewise uses ten fixed initialization seeds and compares
each result with the canonical seed-zero clustering.
