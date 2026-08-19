"""K-Means / t-SNE clustering pipeline.

Usage::

    from raman_analysis.clustering import config, pipeline

    result = pipeline.run(config.OILS)
"""

from . import config, pipeline

__all__ = ["config", "pipeline"]
