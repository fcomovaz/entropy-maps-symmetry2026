"""Portable C99 export, fixed-point mirror, CNN graph compiler, and parity verification."""

from __future__ import annotations

from .config import (
    ALLOWED_NUMERIC_MODES,
    ALLOWED_SHANNON_MODES,
    CImpBuild,
    CImpClassifier,
    CImpCnn,
    CImpConfig,
    CImpDescriptor,
    CImpNumeric,
    CImpPaths,
    CImpTestVectors,
    CImpVerification,
    CImpVisualization,
    DescriptorSpec,
    load_c_imp_config,
)
from .visualize_data import (
    TargetVisualizationBundle,
    VisualizationReport,
    load_target_bundle,
)
from .visualize_plots import generate_c_pipeline_visualizations

__all__ = [
    "ALLOWED_NUMERIC_MODES",
    "ALLOWED_SHANNON_MODES",
    "CImpBuild",
    "CImpClassifier",
    "CImpCnn",
    "CImpConfig",
    "CImpDescriptor",
    "CImpNumeric",
    "CImpPaths",
    "CImpTestVectors",
    "CImpVerification",
    "CImpVisualization",
    "DescriptorSpec",
    "TargetVisualizationBundle",
    "VisualizationReport",
    "generate_c_pipeline_visualizations",
    "load_c_imp_config",
    "load_target_bundle",
]
