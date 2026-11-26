"""Generate publication-ready MDPI methodology and polarity-invariance figures.

Visualizes the end-to-end x94 extraction pipeline:
    preprocessed grayscale image -> B-bin quantization -> nested spatial entropy grid -> x9 asymmetry

And quantifies polarity inversion invariance (I' = 255 - I) across KolektorSDD2,
an MVTec-AD structural object, and an MVTec-AD surface texture.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import tomllib
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.asymmetry_maps import asymmetry_column_names, compute_asymmetry_features
from src.features.entropy_maps import compute_entropy_features, feature_names, quantize_pixels
from src.visualization import PALETTES, apply_style

LOGGER = logging.getLogger("experiments.visualize_methodology")


@dataclass(frozen=True, slots=True)
class DatasetFeatureSpec:
    """Resolved dataset paths and descriptor schema extracted from TOML configs."""

    dataset_key: str
    dataset_name: str
    full_csv_std: Path
    full_csv_inv: Path
    img_root_std: Path
    img_root_inv: Path
    grid_levels: tuple[int, ...]
    bins: int
    asym_scales: tuple[int, ...]
    asym_kinds: tuple[str, ...]
    entropy_cols: tuple[str, ...]
    asym_cols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MethodologyVizConfig:
    """Validated configuration for methodology and polarity-invariance rendering."""

    config_path: Path
    project_root: Path
    kolektor: DatasetFeatureSpec
    mvtec: DatasetFeatureSpec
    output_dir: Path
    journal: str
    palette: str
    dpi: int
    seed: int
    mvtec_object: str
    mvtec_texture: str


@dataclass(frozen=True, slots=True)
class PairVisualSample:
    """Loaded normal/defect pair with raw images, quantized map, 8x8 entropy, and x9 asymmetry."""

    label_tag: str
    normal_img: np.ndarray
    defect_img: np.ndarray
    defect_quant: np.ndarray
    normal_e8: np.ndarray
    defect_e8: np.ndarray
    normal_x9: np.ndarray
    defect_x9: np.ndarray
    asym_names: tuple[str, ...]
    bins: int
    normal_name: str
    defect_name: str
    defect_row: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class InvarianceMetrics:
    """Measured descriptor & pixel differences between standard and polarity-inverted inputs."""

    domain_label: str
    n_matched: int
    mean_pixel_delta: float
    mean_x85_diff: float
    max_x85_diff: float
    mean_x9_diff: float
    max_x9_diff: float
    mean_x94_diff: float
    max_x94_diff: float


def _require(table: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in table:
        raise KeyError(f"Missing required key '{key}' in {where}")
    return table[key]


def _resolve_path(raw: str | Path, project_root: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (project_root / p).resolve()


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"TOML configuration file not found: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_dataset_spec(
    base_data: Mapping[str, Any],
    ent_cfg_path: Path,
    asym_cfg_path: Path,
    project_root: Path,
) -> DatasetFeatureSpec:
    ent_tbl = _require(_load_toml(ent_cfg_path), "entropy_maps", str(ent_cfg_path))
    asym_tbl = _require(_load_toml(asym_cfg_path), "asymmetry_features", str(asym_cfg_path))

    dataset_key = str(_require(ent_tbl, "dataset_key", str(ent_cfg_path)))
    datasets_sec = _require(base_data, "datasets", "configs/base.toml")
    ds_sec = _require(datasets_sec, dataset_key, f"configs/base.toml [datasets.{dataset_key}]")

    reg_e = _require(ent_tbl, "regions", f"{ent_cfg_path} [entropy_maps.regions]")
    out_e = _require(ent_tbl, "output", f"{ent_cfg_path} [entropy_maps.output]")
    reg_a = _require(asym_tbl, "regions", f"{asym_cfg_path} [asymmetry_features.regions]")
    out_a = _require(asym_tbl, "output", f"{asym_cfg_path} [asymmetry_features.output]")

    grid_levels = tuple(int(g) for g in _require(reg_e, "grid_levels", str(ent_cfg_path)))
    bins = int(_require(reg_e, "bins", str(ent_cfg_path)))
    e_tpl = str(_require(out_e, "feature_name_template", str(ent_cfg_path)))

    asym_scales = tuple(int(s) for s in _require(reg_a, "scales", str(asym_cfg_path)))
    asym_kinds = tuple(str(k) for k in _require(reg_a, "kinds", str(asym_cfg_path)))
    col_tpls = dict(_require(out_a, "column_templates", str(asym_cfg_path)))

    e_cols = tuple(feature_names(grid_levels, template=e_tpl))
    a_cols = tuple(asymmetry_column_names(asym_scales, asym_kinds, col_tpls))

    return DatasetFeatureSpec(
        dataset_key=dataset_key,
        dataset_name=str(_require(ds_sec, "name", f"configs/base.toml [datasets.{dataset_key}]")),
        full_csv_std=_resolve_path(_require(ds_sec, "full_csv_path", dataset_key), project_root),
        full_csv_inv=_resolve_path(_require(ds_sec, "full_csv_path_inverted", dataset_key), project_root),
        img_root_std=_resolve_path(_require(ds_sec, "output_path", dataset_key), project_root),
        img_root_inv=_resolve_path(_require(ds_sec, "output_path_inverted", dataset_key), project_root),
        grid_levels=grid_levels,
        bins=bins,
        asym_scales=asym_scales,
        asym_kinds=asym_kinds,
        entropy_cols=e_cols,
        asym_cols=a_cols,
    )


def load_methodology_viz_config(
    config_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> MethodologyVizConfig:
    """Load and validate `[methodology_viz]` TOML and dependent configs."""
    resolved_cfg = config_path if config_path.is_absolute() else (project_root / config_path).resolve()
    raw = _load_toml(resolved_cfg)
    sec = _require(raw, "methodology_viz", str(resolved_cfg))

    base_path = _resolve_path(_require(sec, "base_config", str(resolved_cfg)), project_root)
    base_data = _load_toml(base_path)

    kol_spec = _load_dataset_spec(
        base_data,
        _resolve_path(_require(sec, "kolektor_entropy_config", str(resolved_cfg)), project_root),
        _resolve_path(_require(sec, "kolektor_asymmetry_config", str(resolved_cfg)), project_root),
        project_root,
    )
    mvtec_spec = _load_dataset_spec(
        base_data,
        _resolve_path(_require(sec, "mvtec_entropy_config", str(resolved_cfg)), project_root),
        _resolve_path(_require(sec, "mvtec_asymmetry_config", str(resolved_cfg)), project_root),
        project_root,
    )

    dpi = int(_require(sec, "dpi", str(resolved_cfg)))
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")

    return MethodologyVizConfig(
        config_path=resolved_cfg,
        project_root=project_root,
        kolektor=kol_spec,
        mvtec=mvtec_spec,
        output_dir=_resolve_path(_require(sec, "output_dir", str(resolved_cfg)), project_root),
        journal=str(_require(sec, "journal", str(resolved_cfg))),
        palette=str(_require(sec, "palette", str(resolved_cfg))),
        dpi=dpi,
        seed=int(_require(sec, "seed", str(resolved_cfg))),
        mvtec_object=str(_require(sec, "mvtec_object", str(resolved_cfg))),
        mvtec_texture=str(_require(sec, "mvtec_texture", str(resolved_cfg))),
    )


def _read_csv_rows(csv_path: Path, object_filter: str | None = None) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Required descriptor CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if object_filter is not None:
        rows = [r for r in rows if r.get("object_name") == object_filter]
    return rows


def _image_path_for_row(img_root: Path, row: Mapping[str, str], is_mvtec: bool) -> Path:
    split = row["split"]
    fname = row["file_name"]
    if is_mvtec:
        candidate = img_root / row["object_name"] / split / row["defect_name"] / fname
    else:
        candidate = img_root / split / row["type"] / fname
    if not candidate.is_file():
        raise FileNotFoundError(f"Preprocessed image not found for CSV row: {candidate}")
    return candidate


def _load_grayscale_u8(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def _extract_row_x8_and_x9(
    row: Mapping[str, str],
    spec: DatasetFeatureSpec,
    grid_g: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    e8 = np.array(
        [[float(row[f"E_{grid_g}x{grid_g}_r{r}_c{c}"]) for c in range(grid_g)] for r in range(grid_g)],
        dtype=np.float64,
    )
    x9 = np.array([float(row[col]) for col in spec.asym_cols], dtype=np.float64)
    return e8, x9


def _select_representative_pair(
    spec: DatasetFeatureSpec,
    *,
    is_mvtec: bool,
    object_name: str | None,
    seed: int,
) -> PairVisualSample:
    rows = _read_csv_rows(spec.full_csv_std, object_filter=object_name)
    test_rows = [r for r in rows if r.get("split") == "test"]
    pool = test_rows if test_rows else rows

    good_rows = [r for r in pool if r.get("type") == "good"]
    defect_rows = [r for r in pool if r.get("type") != "good"]
    if not good_rows or not defect_rows:
        raise ValueError(f"Could not find both good and defect samples in {spec.full_csv_std} ({object_name or 'all'})")

    # Deterministic selection: median-like normal sample & highest asymmetry-norm defect sample
    good_norms = np.array(
        [sum(float(r[c]) ** 2 for c in spec.asym_cols) for r in good_rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(good_rows)) if len(good_rows) > 1 else np.array([0])
    good_sorted = sorted(order.tolist(), key=lambda i: float(good_norms[i]))
    good_row = good_rows[good_sorted[len(good_sorted) // 2]]

    def_row = max(
        defect_rows,
        key=lambda r: (sum(float(r[c]) ** 2 for c in spec.asym_cols), r.get("file_name", "")),
    )

    good_img_path = _image_path_for_row(spec.img_root_std, good_row, is_mvtec=is_mvtec)
    def_img_path = _image_path_for_row(spec.img_root_std, def_row, is_mvtec=is_mvtec)
    good_img = _load_grayscale_u8(good_img_path)
    def_img = _load_grayscale_u8(def_img_path)
    def_quant = quantize_pixels(def_img, spec.bins)

    norm_e8, norm_x9 = _extract_row_x8_and_x9(good_row, spec, grid_g=8)
    def_e8, def_x9 = _extract_row_x8_and_x9(def_row, spec, grid_g=8)

    label_tag = f"{spec.dataset_name} ({object_name})" if object_name else spec.dataset_name
    return PairVisualSample(
        label_tag=label_tag,
        normal_img=good_img,
        defect_img=def_img,
        defect_quant=def_quant,
        normal_e8=norm_e8,
        defect_e8=def_e8,
        normal_x9=norm_x9,
        defect_x9=def_x9,
        asym_names=spec.asym_cols,
        bins=spec.bins,
        normal_name=good_row["file_name"],
        defect_name=def_row["file_name"],
        defect_row=def_row,
    )


def _draw_grid_overlay(ax: plt.Axes, height: int, width: int, g: int = 8) -> None:
    for k in range(1, g):
        y = k * height / g - 0.5
        x = k * width / g - 0.5
        ax.axhline(y, color="#FFFFFF", linewidth=0.55, alpha=0.65)
        ax.axvline(x, color="#FFFFFF", linewidth=0.55, alpha=0.65)


def _plot_x94_methodology_figure(
    sample: PairVisualSample,
    out_path: Path,
    cfg: MethodologyVizConfig,
) -> Path:
    """Render 2x3 MDPI figure walking through image -> quantization -> 8x8 entropy -> x9 asymmetry."""
    apply_style(cfg.journal, palette=cfg.palette)
    pal = PALETTES[cfg.palette]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), dpi=cfg.dpi)
    fig.suptitle(
        f"Multiscale Entropy & Asymmetry Descriptor (x94) Pipeline — {sample.label_tag}",
        fontsize=12.5,
        fontweight="semibold",
    )

    h, w = sample.normal_img.shape
    vmax_e = float(max(np.max(sample.normal_e8), np.max(sample.defect_e8), 1e-6))

    # (0, 0) Normal Grayscale + 8x8 partition
    axes[0, 0].imshow(sample.normal_img, cmap="gray", vmin=0, vmax=255)
    _draw_grid_overlay(axes[0, 0], h, w, g=8)
    axes[0, 0].set_title(f"(a) Normal Input + 8x8 Grid ({sample.normal_name})")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # (0, 1) Defect Grayscale + 8x8 partition
    axes[0, 1].imshow(sample.defect_img, cmap="gray", vmin=0, vmax=255)
    _draw_grid_overlay(axes[0, 1], h, w, g=8)
    axes[0, 1].set_title(f"(b) Defect Input + 8x8 Grid ({sample.defect_name})")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # (0, 2) B-bin Quantized Defect Map
    im_q = axes[0, 2].imshow(sample.defect_quant, cmap="cividis", vmin=0, vmax=sample.bins - 1)
    _draw_grid_overlay(axes[0, 2], h, w, g=8)
    axes[0, 2].set_title(f"(c) Quantized Defect Map ({sample.bins} bins)")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    fig.colorbar(im_q, ax=axes[0, 2], fraction=0.046, pad=0.04, label="Bin Index")

    # (1, 0) Normal 8x8 Entropy Map
    im_en = axes[1, 0].imshow(sample.normal_e8, cmap="viridis", vmin=0.0, vmax=vmax_e)
    axes[1, 0].set_title("(d) Normal E_8x8 Entropy Map")
    axes[1, 0].set_xlabel("Grid Column c")
    axes[1, 0].set_ylabel("Grid Row r")
    fig.colorbar(im_en, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Normalized H")

    # (1, 1) Defect 8x8 Entropy Map
    im_ed = axes[1, 1].imshow(sample.defect_e8, cmap="viridis", vmin=0.0, vmax=vmax_e)
    axes[1, 1].set_title("(e) Defect E_8x8 Entropy Map")
    axes[1, 1].set_xlabel("Grid Column c")
    axes[1, 1].set_ylabel("Grid Row r")
    fig.colorbar(im_ed, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Normalized H")

    # (1, 2) x9 Asymmetry Profile (Normal vs Defect)
    ax_x9 = axes[1, 2]
    x_idx = np.arange(len(sample.asym_names), dtype=np.float64)
    width = 0.36
    ax_x9.bar(x_idx - width / 2.0, sample.normal_x9, width=width, color=pal[0], label="Normal (good)")
    ax_x9.bar(x_idx + width / 2.0, sample.defect_x9, width=width, color=pal[1], label="Defect")
    short_names = [n.replace("asym_", "") for n in sample.asym_names]
    ax_x9.set_xticks(x_idx)
    ax_x9.set_xticklabels(short_names, rotation=32, ha="right", fontsize=8)
    ax_x9.set_ylabel("Asymmetry Score A_s")
    ax_x9.set_title("(f) Multiscale Symmetry Breaking (x9)")
    ax_x9.grid(axis="y", linestyle="--", alpha=0.35)
    ax_x9.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _row_key(row: Mapping[str, str], is_mvtec: bool) -> tuple[str, ...]:
    if is_mvtec:
        return (row["object_name"], row["split"], row["type"], row["defect_name"], row["file_name"])
    return (row["split"], row["type"], row["file_name"])


def _compute_invariance_stats(
    spec: DatasetFeatureSpec,
    *,
    is_mvtec: bool,
    object_name: str | None,
    domain_label: str,
    sample_row_std: Mapping[str, str],
) -> tuple[InvarianceMetrics, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    std_rows = _read_csv_rows(spec.full_csv_std, object_filter=object_name)
    inv_rows = _read_csv_rows(spec.full_csv_inv, object_filter=object_name)
    inv_by_key = {_row_key(r, is_mvtec=is_mvtec): r for r in inv_rows}

    all_cols = spec.entropy_cols + spec.asym_cols
    n_e = len(spec.entropy_cols)
    std_vecs: list[list[float]] = []
    inv_vecs: list[list[float]] = []

    for r_std in std_rows:
        r_inv = inv_by_key.get(_row_key(r_std, is_mvtec=is_mvtec))
        if r_inv is None:
            continue
        std_vecs.append([float(r_std[c]) for c in all_cols])
        inv_vecs.append([float(r_inv[c]) for c in all_cols])

    if not std_vecs:
        raise ValueError(f"No matched standard/inverted CSV rows found for {domain_label}")

    mat_std = np.array(std_vecs, dtype=np.float64)
    mat_inv = np.array(inv_vecs, dtype=np.float64)
    diff = np.abs(mat_inv - mat_std)

    key = _row_key(sample_row_std, is_mvtec=is_mvtec)
    sample_row_inv = inv_by_key.get(key)
    if sample_row_inv is None:
        raise KeyError(f"Matched inverted row missing for key={key} in {spec.full_csv_inv}")

    img_std = _load_grayscale_u8(_image_path_for_row(spec.img_root_std, sample_row_std, is_mvtec=is_mvtec))
    img_inv = _load_grayscale_u8(_image_path_for_row(spec.img_root_inv, sample_row_inv, is_mvtec=is_mvtec))
    mean_px_delta = float(np.mean(np.abs(img_inv.astype(np.float64) - img_std.astype(np.float64))) / 255.0)

    metrics = InvarianceMetrics(
        domain_label=domain_label,
        n_matched=int(mat_std.shape[0]),
        mean_pixel_delta=mean_px_delta,
        mean_x85_diff=float(np.mean(diff[:, :n_e])),
        max_x85_diff=float(np.max(diff[:, :n_e])),
        mean_x9_diff=float(np.mean(diff[:, n_e:])),
        max_x9_diff=float(np.max(diff[:, n_e:])),
        mean_x94_diff=float(np.mean(diff)),
        max_x94_diff=float(np.max(diff)),
    )
    e8_std, _ = _extract_row_x8_and_x9(sample_row_std, spec, grid_g=8)
    e8_inv, _ = _extract_row_x8_and_x9(sample_row_inv, spec, grid_g=8)
    return metrics, img_std, img_inv, e8_std, e8_inv


def _plot_polarity_invariance_figure(
    domains: Sequence[InvarianceMetrics],
    kol_img_std: np.ndarray,
    kol_img_inv: np.ndarray,
    kol_e8_std: np.ndarray,
    kol_e8_inv: np.ndarray,
    bins: int,
    out_path: Path,
    cfg: MethodologyVizConfig,
) -> Path:
    """Render 2x3 MDPI figure proving x94 invariance under I' = 255 - I."""
    apply_style(cfg.journal, palette=cfg.palette)
    pal = PALETTES[cfg.palette]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6), dpi=cfg.dpi)
    fig.suptitle(
        "Polarity Inversion Invariance of x94 (I' = 255 - I) Across Industrial Vision Domains",
        fontsize=12.5,
        fontweight="semibold",
    )

    # Top row: Visual demonstration on representative defect image
    axes[0, 0].imshow(kol_img_std, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("(a) Standard Input I")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    axes[0, 1].imshow(kol_img_inv, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("(b) Polarity-Inverted I' = 255 - I")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # Exact bin-count reflection verification for single image
    q_std = quantize_pixels(kol_img_std, bins)
    q_inv = quantize_pixels(kol_img_inv, bins)
    hist_std = np.bincount(q_std.ravel(), minlength=bins).astype(np.float64)
    hist_inv = np.bincount(q_inv.ravel(), minlength=bins).astype(np.float64)
    hist_std_pct = 100.0 * hist_std / max(1.0, hist_std.sum())
    hist_inv_pct = 100.0 * hist_inv / max(1.0, hist_inv.sum())
    b_idx = np.arange(bins, dtype=np.float64)
    axes[0, 2].bar(b_idx - 0.18, hist_std_pct, width=0.35, color=pal[0], label="Standard I")
    axes[0, 2].bar(b_idx + 0.18, hist_inv_pct, width=0.35, color=pal[1], label="Inverted I'")
    axes[0, 2].set_xlabel(f"Quantized Bin b in [0, {bins - 1}]")
    axes[0, 2].set_ylabel("Pixel Share (%)")
    axes[0, 2].set_title("(c) Bin Permutation Under Intensity Inversion")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(axis="y", linestyle="--", alpha=0.35)

    # Bottom-left & Bottom-center: 8x8 Entropy Maps Standard vs Inverted
    vmax_e = float(max(np.max(kol_e8_std), np.max(kol_e8_inv), 1e-6))
    im_s = axes[1, 0].imshow(kol_e8_std, cmap="viridis", vmin=0.0, vmax=vmax_e)
    axes[1, 0].set_title("(d) E_8x8(I) Entropy Map")
    axes[1, 0].set_xlabel("Grid Column c")
    axes[1, 0].set_ylabel("Grid Row r")
    fig.colorbar(im_s, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_i = axes[1, 1].imshow(kol_e8_inv, cmap="viridis", vmin=0.0, vmax=vmax_e)
    max_single_diff = float(np.max(np.abs(kol_e8_inv - kol_e8_std)))
    axes[1, 1].set_title(f"(e) E_8x8(I') Map (max|diff|={max_single_diff:.2e})")
    axes[1, 1].set_xlabel("Grid Column c")
    axes[1, 1].set_ylabel("Grid Row r")
    fig.colorbar(im_i, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Bottom-right: Multi-domain quantitative invariance summary
    ax_q = axes[1, 2]
    d_names = [f"{d.domain_label}\n(n={d.n_matched})" for d in domains]
    x_d = np.arange(len(domains), dtype=np.float64)
    w = 0.25
    px_shift = [max(d.mean_pixel_delta, 1e-9) for d in domains]
    x85_shift = [max(d.max_x85_diff, 1e-9) for d in domains]
    x9_shift = [max(d.max_x9_diff, 1e-9) for d in domains]

    ax_q.bar(x_d - w, px_shift, width=w, color=pal[1], label="Mean Pixel |I' - I| / 255")
    ax_q.bar(x_d, x85_shift, width=w, color=pal[0], label="Max |x85(I') - x85(I)|")
    ax_q.bar(x_d + w, x9_shift, width=w, color=pal[2], label="Max |x9(I') - x9(I)|")
    ax_q.set_yscale("log")
    ax_q.set_ylim(1e-10, 5.0)
    for idx_d, d in enumerate(domains):
        ax_q.text(
            x_d[idx_d],
            max(px_shift[idx_d], 1e-9) * 1.35,
            f"d_px={d.mean_pixel_delta:.2f}\nd_x94={d.max_x94_diff:.1e}",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
    ax_q.set_xticks(x_d)
    ax_q.set_xticklabels(d_names, fontsize=8)
    ax_q.set_ylabel("Absolute Difference (log scale)")
    ax_q.set_title("(f) Measured Pixel Shift vs x94 Invariance")
    ax_q.legend(loc="center right", fontsize=7.5)
    ax_q.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_methodology_visualization(
    cfg: MethodologyVizConfig,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Generate all three methodology & polarity-invariance publication figures."""
    kol_fig = cfg.output_dir / "kolektorsdd2_x94_methodology.png"
    mvtec_fig = cfg.output_dir / f"mvtec_ad_{cfg.mvtec_object}_x94_methodology.png"
    inv_fig = cfg.output_dir / "polarity_inversion_invariance_comparison.png"
    expected = (kol_fig, mvtec_fig, inv_fig)

    if dry_run:
        LOGGER.info(
            "Dry-run OK | dpi=%d | journal=%s | palette=%s | mvtec_object=%s | mvtec_texture=%s | outputs=%s",
            cfg.dpi,
            cfg.journal,
            cfg.palette,
            cfg.mvtec_object,
            cfg.mvtec_texture,
            [str(p) for p in expected],
        )
        return expected

    if all(p.is_file() for p in expected) and not overwrite:
        LOGGER.info("All methodology figures already exist in %s; skipping (use --overwrite).", cfg.output_dir)
        return expected

    kol_pair = _select_representative_pair(cfg.kolektor, is_mvtec=False, object_name=None, seed=cfg.seed)
    mvtec_pair = _select_representative_pair(cfg.mvtec, is_mvtec=True, object_name=cfg.mvtec_object, seed=cfg.seed)
    mvtec_tex_pair = _select_representative_pair(cfg.mvtec, is_mvtec=True, object_name=cfg.mvtec_texture, seed=cfg.seed)

    # Touch compute_entropy_features / compute_asymmetry_features on representative sample as sanity check
    e_chk = compute_entropy_features(
        kol_pair.defect_img,
        grid_levels=cfg.kolektor.grid_levels,
        bins=cfg.kolektor.bins,
        backend="numpy_baseline",
    )
    if e_chk.shape != (1, len(cfg.kolektor.entropy_cols)):
        raise ValueError(f"Unexpected runtime entropy shape {e_chk.shape}")
    e_by_scale = {
        2: e_chk[:, 1:5],
        4: e_chk[:, 5:21],
        8: e_chk[:, 21:85],
    }
    _ = compute_asymmetry_features(
        e_by_scale,
        scales=cfg.kolektor.asym_scales,
        kinds=cfg.kolektor.asym_kinds,
    )

    _plot_x94_methodology_figure(kol_pair, kol_fig, cfg)
    _plot_x94_methodology_figure(mvtec_pair, mvtec_fig, cfg)

    kol_inv_m, kol_img_std, kol_img_inv, kol_e8_std, kol_e8_inv = _compute_invariance_stats(
        cfg.kolektor,
        is_mvtec=False,
        object_name=None,
        domain_label="KolektorSDD2",
        sample_row_std=kol_pair.defect_row,
    )
    obj_inv_m, _, _, _, _ = _compute_invariance_stats(
        cfg.mvtec,
        is_mvtec=True,
        object_name=cfg.mvtec_object,
        domain_label=f"MVTec ({cfg.mvtec_object})",
        sample_row_std=mvtec_pair.defect_row,
    )
    tex_inv_m, _, _, _, _ = _compute_invariance_stats(
        cfg.mvtec,
        is_mvtec=True,
        object_name=cfg.mvtec_texture,
        domain_label=f"MVTec ({cfg.mvtec_texture})",
        sample_row_std=mvtec_tex_pair.defect_row,
    )

    _plot_polarity_invariance_figure(
        (kol_inv_m, obj_inv_m, tex_inv_m),
        kol_img_std,
        kol_img_inv,
        kol_e8_std,
        kol_e8_inv,
        bins=cfg.kolektor.bins,
        out_path=inv_fig,
        cfg=cfg,
    )
    LOGGER.info("Generated methodology figures in %s", cfg.output_dir)
    return expected


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for methodology visualization."""
    parser = argparse.ArgumentParser(
        description="Generate MDPI methodology and polarity-invariance figures.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/visualization/methodology.toml",
        help="Path to TOML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and log planned outputs without creating files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing output PNG figures.",
    )
    return parser


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """CLI entrypoint for methodology figure generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_methodology_viz_config(Path(args.config), project_root=project_root)
        run_methodology_visualization(cfg, dry_run=bool(args.dry_run), overwrite=bool(args.overwrite))
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        LOGGER.error("Methodology visualization failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
