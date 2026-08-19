"""Write the exact library versions a run used, for reproducibility.

This has nothing to do with clustering itself; it is recorded once
(from the oils K-Means script - see ``scripts/run_kmeans_oils.py``)
alongside the clustering figures because that is where the original
notebook first captured it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd
import numpy as np
import plotly
import seaborn as sns
import sklearn


def write_environment_report(out_path: Path) -> None:
    versions = {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
        "plotly": plotly.__version__,
        "scikit-learn": sklearn.__version__,
    }

    print("=" * 50)
    print(f"Python Version      : {sys.version}")
    print(f"Pandas Version      : {versions['pandas']}")
    print(f"NumPy Version       : {versions['numpy']}")
    print(f"Matplotlib Version  : {versions['matplotlib']}")
    print(f"Seaborn Version     : {versions['seaborn']}")
    print(f"Plotly Version      : {versions['plotly']}")
    print(f"Scikit-learn Version: {versions['scikit-learn']}")
    print("=" * 50)

    lines = [
        "# Project Requirements", "",
        "## Environment", "", f"- Python: {sys.version.split()[0]}", "",
        "## Libraries", "",
    ]
    lines += [f"- {name}=={version}" for name, version in versions.items()]
    out_path.write_text("\n".join(lines) + "\n")

    print("requirements.md has been created successfully.")
