"""Publication-grade scientific plotting package (MDPI / Springer styles, Agg-safe)."""

from __future__ import annotations

from .sci_plots import (
    FIG_SIZES,
    HATCHES,
    LINESTYLES,
    MARKERS,
    PALETTES,
    annotate_points_non_overlapping,
    apply_style,
    create_grid,
    get_colors,
    plot_bars,
    plot_donut,
    plot_heatmap,
    plot_histogram,
    plot_lines,
    plot_regression,
    plot_scatter,
    save_figure,
)

__all__ = [
    "FIG_SIZES",
    "HATCHES",
    "LINESTYLES",
    "MARKERS",
    "PALETTES",
    "annotate_points_non_overlapping",
    "apply_style",
    "create_grid",
    "get_colors",
    "plot_bars",
    "plot_donut",
    "plot_heatmap",
    "plot_histogram",
    "plot_lines",
    "plot_regression",
    "plot_scatter",
    "save_figure",
]
