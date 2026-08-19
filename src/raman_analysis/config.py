"""Constants shared by both the clustering and decision-tree pipelines.

Everything here is data, not behaviour: oil class labels, the wavenumber
region break used by every broken-axis spectral plot, and the two color
palettes used to keep a given oil's color consistent across every figure
in the project (pure-oil plots use the plain oil code; chip plots use the
same code suffixed with " C", because the chip datasets label oil type as
e.g. "GNO C" rather than "GNO").
"""

from __future__ import annotations

# Random seed for every Decision Tree, train/test split, and CV fold.
# Matches ``random_state=1`` throughout the original notebooks.
DECISION_TREE_RANDOM_STATE = 1

# Random seed for StandardScaler-based clustering, t-SNE, and KMeans.
# Matches ``RS = 0`` in the original K-Means notebooks.
CLUSTERING_RANDOM_STATE = 0

# Five edible oils studied throughout the project, in the fixed display
# order used by every legend and confusion-matrix axis in the notebooks.
OIL_CODES = ["SO", "PO", "GNO", "SOYO", "VO"]
OIL_CODES_SORTED = ["GNO", "PO", "SO", "SOYO", "VO"]

# Wavenumber (cm^-1) below which the "left" half of a broken-axis spectral
# plot is drawn; the "right" half starts at RIGHT_REGION_MIN. The gap
# between them (the fingerprint-to-CH-stretch silent region) is what gets
# visually broken out of every mean-spectrum and cluster-profile plot.
LEFT_REGION_MAX = 1900
RIGHT_REGION_MIN = 2600

# Oil -> color, for pure-oil plots (t-SNE, cluster profiles, mean spectra).
OIL_PALETTE = {
    "SO": "tab:blue",
    "PO": "tab:orange",
    "GNO": "tab:green",
    "SOYO": "tab:red",
    "VO": "tab:purple",
}

# Same oils, flat (non-"tab:") colors - used for the 3D plotly figure and
# the cluster-profile line plots, matching the original notebooks exactly.
OIL_PALETTE_FLAT = {
    "SO": "blue",
    "PO": "orange",
    "GNO": "green",
    "SOYO": "red",
    "VO": "purple",
}

# Chip samples are labelled "<oil code> C" (e.g. "GNO C"); same colors.
CHIPS_OIL_PALETTE = {f"{code} C": color for code, color in OIL_PALETTE.items()}
CHIPS_OIL_PALETTE_FLAT = {
    f"{code} C": color for code, color in OIL_PALETTE_FLAT.items()
}
