"""Feature extraction modules for informative regions, visual entropy, and descriptors."""
from __future__ import annotations

from src.features.entropy_csv import (
    EntropyCsvConfig,
    EntropyExportSummary,
    export_entropy_csv,
    load_entropy_config,
    validate_header,
)
from src.features.entropy_maps import (
    EntropyPlan,
    RegionSpec,
    build_entropy_plan,
    build_quantization_lut,
    build_shannon_term_lut,
    compute_entropy_features,
    feature_names,
    normalized_entropy,
    quantize_pixels,
    region_edges,
    region_histograms,
    validate_bins,
    validate_grid_levels,
)

__all__ = [
    "EntropyCsvConfig",
    "EntropyExportSummary",
    "EntropyPlan",
    "RegionSpec",
    "build_entropy_plan",
    "build_quantization_lut",
    "build_shannon_term_lut",
    "compute_entropy_features",
    "export_entropy_csv",
    "feature_names",
    "load_entropy_config",
    "normalized_entropy",
    "quantize_pixels",
    "region_edges",
    "region_histograms",
    "validate_bins",
    "validate_grid_levels",
    "validate_header",
]
