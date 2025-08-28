"""
Unified Tissue Core Alignment Pipeline

Integrates tissue core segmentation with VALIS registration according to 
the specification in docs/pipeline_plan.md.
"""

from .pipeline import UnifiedPipeline
from .config import UnifiedConfig

__version__ = "1.0.0"
__all__ = ["UnifiedPipeline", "UnifiedConfig"]
