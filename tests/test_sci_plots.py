"""Unit tests for the headless-safe MDPI/Springer `src.visualization` plotting package."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import (
    annotate_points_non_overlapping,
    apply_style,
    create_grid,
    plot_bars,
    plot_donut,
    plot_heatmap,
    plot_histogram,
    plot_lines,
    plot_regression,
    plot_scatter,
    save_figure,
)


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1] or a[3] <= b[2] or a[2] >= b[3])


class TestSciPlots(unittest.TestCase):
    """Verify MDPI styling, Agg safety, layout cleanliness, and collision-free annotations."""

    def tearDown(self) -> None:
        plt.close("all")

    def test_headless_agg_and_style_presets(self) -> None:
        self.assertEqual(mpl.get_backend().lower(), "agg")

        apply_style("mdpi", palette="editorial")
        self.assertEqual(int(mpl.rcParams["savefig.dpi"]), 300)
        self.assertEqual(int(mpl.rcParams["pdf.fonttype"]), 42)
        self.assertFalse(bool(mpl.rcParams["axes.spines.top"]))
        self.assertFalse(bool(mpl.rcParams["axes.spines.right"]))
        self.assertEqual(list(mpl.rcParams["font.family"]), ["sans-serif"])

        apply_style("springer", palette="classic", font_size=10.5)
        self.assertTrue(bool(mpl.rcParams["axes.spines.top"]))
        self.assertEqual(mpl.rcParams["xtick.direction"], "in")
        self.assertEqual(float(mpl.rcParams["font.size"]), 10.5)

        with self.assertRaises(ValueError):
            apply_style("unknown_journal")
        with self.assertRaises(ValueError):
            apply_style("mdpi", palette="unknown_palette")

    def test_create_grid_and_subplot_composition_no_layout_warning(self) -> None:
        apply_style("mdpi", palette="editorial")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fig, axes = create_grid(1, 2, figsize="double")
                plot_scatter(
                    {
                        "lr_dot": (0.30, 0.80),
                        "mlp_16_dot": (0.38, 0.82),
                        "mobilenet_like": (4.13, 0.76),
                    },
                    xlabel="CPU Latency (ms, log scale)",
                    ylabel="F1-Score",
                    xscale="log",
                    annotate_points=True,
                    pareto_direction=("min", "max"),
                    ax=axes[0, 0],
                )
                plot_heatmap(
                    np.array([[0.10, 0.25], [0.40, 0.85]]),
                    cmap="cividis",
                    cbar_label="Entropy",
                    ax=axes[0, 1],
                )
                png_out = save_figure(fig, tmp_dir / "grid_test.png", dpi=300)
                pdf_out = save_figure(fig, tmp_dir / "grid_test.pdf", dpi=300)
                plt.close(fig)

            layout_warns = [
                str(w.message)
                for w in caught
                if "tight_layout" in str(w.message).lower()
                or "constrained_layout" in str(w.message).lower()
            ]
            self.assertEqual(layout_warns, [])
            self.assertTrue(png_out.is_file())
            self.assertTrue(pdf_out.is_file())
            self.assertGreater(png_out.stat().st_size, 1000)
            self.assertGreater(pdf_out.stat().st_size, 1000)

    def test_non_overlapping_annotations_on_clustered_log_pareto(self) -> None:
        apply_style("mdpi", palette="editorial")
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        ax.set_xscale("log")
        labels = [
            "lr_dot",
            "mlp_16_dot",
            "ultra_tiny",
            "ds_cnn",
            "tiny_cnn",
            "mobilenet_like",
            "tiny_resnet",
            "small_cnn",
            "torchvision_mobilenet",
        ]
        xs = [0.018, 0.019, 0.30, 0.31, 37.9, 16.8, 45.4, 52.2, 0.95]
        ys = [0.425, 0.489, 0.3611, 0.3611, 0.619, 0.622, 0.710, 0.596, 0.3611]
        ax.scatter(xs, ys)
        ax.set_xlim(0.008, 120.0)
        ax.set_ylim(0.28, 0.80)

        reserved = [(0.62, 0.98, 0.02, 0.24)]
        placed_boxes = annotate_points_non_overlapping(
            ax,
            labels,
            xs,
            ys,
            xscale="log",
            reserved_boxes_axes=reserved,
        )

        self.assertEqual(len(placed_boxes), len(labels))
        self.assertEqual(len(ax.texts), len(labels))
        for box in placed_boxes:
            x0, x1, y0, y1 = box
            self.assertGreaterEqual(x0, 0.0)
            self.assertLessEqual(x1, 1.0)
            self.assertGreaterEqual(y0, 0.0)
            self.assertLessEqual(y1, 1.0)
            self.assertFalse(_boxes_overlap(box, reserved[0]), f"Label box {box} overlapped legend")

        for i in range(len(placed_boxes)):
            for j in range(i + 1, len(placed_boxes)):
                self.assertFalse(
                    _boxes_overlap(placed_boxes[i], placed_boxes[j]),
                    f"Label {labels[i]} {placed_boxes[i]} overlaps {labels[j]} {placed_boxes[j]}",
                )

    def test_all_master_plotters_smoke(self) -> None:
        apply_style("mdpi", palette="editorial")
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            x = np.arange(5, dtype=float)
            fig, _ = plot_lines(
                {
                    "Simple": x * 0.1,
                    "Symmetric": (x, x * 0.1 + 0.2, np.full(5, 0.03)),
                    "Asymmetric": (x, x * 0.1 + 0.4, x * 0.1 + 0.35, x * 0.1 + 0.48),
                },
                xlabel="Step",
                ylabel="Score",
                save_path=tmp_dir / "lines.png",
            )
            plt.close(fig)

            xr = np.linspace(1.0, 10.0, 15)
            fig, _ = plot_regression(
                {"Series A": (xr, 0.2 + 0.05 * xr + rng.normal(0, 0.01, size=15))},
                show_r2=True,
                show_eq=True,
                show_ci=True,
                save_path=tmp_dir / "reg.png",
            )
            plt.close(fig)

            fig, _ = plot_bars(
                {"A": 0.72, "B": 0.85, "C": 0.64},
                orientation="horizontal",
                save_path=tmp_dir / "bars_single.png",
            )
            plt.close(fig)

            fig, _ = plot_bars(
                {"x85": [0.71, 0.43], "x94": [0.80, 0.58]},
                categories=["Bottle", "Kolektor"],
                errors={"x85": [0.02, 0.03], "x94": [0.01, 0.02]},
                save_path=tmp_dir / "bars_grouped.png",
            )
            plt.close(fig)

            fig, _ = plot_histogram(
                {"Good": rng.normal(0.2, 0.05, 60), "Defect": rng.normal(0.6, 0.08, 60)},
                density=True,
                save_path=tmp_dir / "hist.png",
            )
            plt.close(fig)

            fig, _ = plot_donut(
                {"x85": 70.0, "x9": 30.0},
                center_text="100%",
                save_path=tmp_dir / "donut.png",
            )
            plt.close(fig)

            fig, _ = plot_heatmap(
                rng.uniform(-0.3, 0.3, size=(4, 4)),
                cmap="RdBu_r",
                center=0.0,
                save_path=tmp_dir / "heatmap.png",
            )
            plt.close(fig)

            for fname in (
                "lines.png",
                "reg.png",
                "bars_single.png",
                "bars_grouped.png",
                "hist.png",
                "donut.png",
                "heatmap.png",
            ):
                fpath = tmp_dir / fname
                self.assertTrue(fpath.is_file(), f"Missing {fname}")
                self.assertGreater(fpath.stat().st_size, 1000, f"Too small {fname}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
