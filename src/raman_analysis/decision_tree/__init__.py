"""Decision Tree classification pipeline (baseline / pre- / post-pruning).

Usage::

    from raman_analysis.decision_tree import config, pipeline

    pipeline.run(config.OILS)
"""

from . import config, pipeline

__all__ = ["config", "pipeline"]
