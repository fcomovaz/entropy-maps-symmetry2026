"""Test vector exporter: preprocessed PNG/partition -> raw .bin + manifest + golden x94 CSVs + MCU C headers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.features.asymmetry_maps import asymmetry_column_names, entropy_column_names
from src.models.cnn.data import CnnDatasetPartition, load_cnn_partition

from .c_writer import format_c_array, sha256_bytes, write_header
from .config import CImpConfig
from .descriptor_reference import extract_x94_f64_oracle, extract_x94_q15_batch


@dataclass(frozen=True, slots=True)
class ExportedTestVectors:
    """In-memory and on-disk handle to exported evaluation images and golden x94 tables."""

    vectors_dir: Path
    images_root: Path
    manifest_path: Path
    golden_x94_f64_path: Path
    golden_x94_q15_path: Path
    indices: np.ndarray
    file_names: tuple[str, ...]
    labels: np.ndarray
    images_u8: np.ndarray          # [N, H, W] uint8
    images_f32: np.ndarray         # [N, 1, H, W] float32 (normalized)
    x94_f64: np.ndarray            # [N, 94] float64
    x94_q15: np.ndarray            # [N, 94] int16
    embedded_headers: tuple[Path, ...]


def x94_column_names(cfg: CImpConfig) -> list[str]:
    """Return ordered canonical x94 column names from referenced asymmetry/classification config."""
    cols: list[str] = []
    tpl = cfg.asymmetry_cfg.feature_name_template
    for g in cfg.descriptor_spec.grid_levels:
        cols.extend(entropy_column_names(g, tpl))
    cols.extend(
        asymmetry_column_names(
            cfg.descriptor_spec.asym_scales,
            cfg.descriptor_spec.asym_kinds,
            cfg.asymmetry_cfg.column_templates,
        )
    )
    return cols


def _select_indices(y: np.ndarray, max_images: int, seed: int) -> np.ndarray:
    """Return full test indices or deterministic stratified subsample of size `max_images`."""
    n = int(y.shape[0])
    if max_images <= 0 or max_images >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    n1 = max(1, min(len(idx1), round(max_images * len(idx1) / max(1, n)))) if len(idx1) > 0 else 0
    n0 = max_images - n1
    if n0 > len(idx0):
        n0 = len(idx0)
        n1 = min(len(idx1), max_images - n0)
    sel0 = rng.choice(idx0, size=n0, replace=False) if n0 > 0 else np.empty(0, dtype=np.int64)
    sel1 = rng.choice(idx1, size=n1, replace=False) if n1 > 0 else np.empty(0, dtype=np.int64)
    combined = np.sort(np.concatenate([sel0, sel1])).astype(np.int64)
    return combined


def _partition_to_u8(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Losslessly invert normalized float32 tensor [N, 1, H, W] back to row-major uint8 [N, H, W]."""
    arr = np.asarray(X, dtype=np.float64)[:, 0, :, :]
    raw = np.rint((arr * std + mean) * 255.0)
    return np.ascontiguousarray(np.clip(raw, 0.0, 255.0), dtype=np.uint8)


def export_test_vectors(
    cfg: CImpConfig,
    partition: CnnDatasetPartition | None = None,
    max_images_override: int | None = None,
) -> ExportedTestVectors:
    """Serialize evaluation images as raw .bin + manifest + golden x94 CSVs + embedded C headers."""
    part = (
        partition
        if partition is not None
        else load_cnn_partition(cfg.cnn_cfg, object_name=cfg.object_name, seed=cfg.seed)
    )
    max_imgs = cfg.test_vectors.max_images if max_images_override is None else max_images_override
    y_full = part.y_test.cpu().numpy().astype(np.int64)
    sel = _select_indices(y_full, max_imgs, seed=cfg.seed)

    X_f32_sel = part.X_test[sel].cpu().numpy().astype(np.float32)
    y_sel = y_full[sel]
    fnames_sel = tuple(part.test_file_names[int(i)] for i in sel)
    u8_sel = _partition_to_u8(X_f32_sel, cfg.cnn_cfg.data.normalize_mean, cfg.cnn_cfg.data.normalize_std)

    vectors_dir = cfg.results_dir / "vectors"
    images_root = vectors_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    cfg.generated_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = vectors_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["index", "file_name", "split", "label", "bin_relpath", "sha256"])
        for row_idx in range(len(sel)):
            raw_bytes = np.ascontiguousarray(u8_sel[row_idx], dtype=np.uint8).tobytes(order="C")
            rel_name = f"{row_idx:05d}.bin"
            (images_root / rel_name).write_bytes(raw_bytes)
            writer.writerow(
                [
                    row_idx,
                    fnames_sel[row_idx],
                    cfg.test_vectors.split,
                    int(y_sel[row_idx]),
                    rel_name,
                    sha256_bytes(raw_bytes),
                ]
            )

    # Golden x94 float64 (src/) and bit-exact Q15 mirror
    x94_f64 = extract_x94_f64_oracle(u8_sel, cfg)
    x94_q15 = extract_x94_q15_batch(u8_sel, cfg)
    feat_cols = x94_column_names(cfg)

    golden_f64_path = vectors_dir / "golden_x94_f64.csv"
    with golden_f64_path.open("w", newline="", encoding="utf-8") as f64f:
        w = csv.writer(f64f)
        w.writerow(["index", "label", *feat_cols])
        for row_idx in range(len(sel)):
            w.writerow([row_idx, int(y_sel[row_idx]), *(f"{float(v):.10g}" for v in x94_f64[row_idx])])

    golden_q15_path = vectors_dir / "golden_x94_q15.csv"
    with golden_q15_path.open("w", newline="", encoding="utf-8") as q15f:
        w = csv.writer(q15f)
        w.writerow(["index", "label", *feat_cols])
        for row_idx in range(len(sel)):
            w.writerow([row_idx, int(y_sel[row_idx]), *(int(v) for v in x94_q15[row_idx])])

    embedded: list[Path] = []
    n_embed = min(cfg.test_vectors.embed_headers_count, len(sel))
    for k in range(n_embed):
        hdr_path = cfg.generated_dir / f"test_image_{k}.h"
        flat = np.ascontiguousarray(u8_sel[k], dtype=np.uint8).ravel()
        body = "\n".join(
            [
                f"#define CIMP_TEST_IMAGE_{k}_LABEL {int(y_sel[k])}",
                f'#define CIMP_TEST_IMAGE_{k}_FILE "{fnames_sel[k]}"',
                format_c_array(f"CIMP_TEST_IMAGE_{k}_U8", flat, "uint8_t", per_line=16, hex_int8=True),
            ]
        )
        write_header(
            hdr_path,
            f"CIMP_TEST_IMAGE_{k}_H",
            ["<stdint.h>"],
            body,
            [f"Sample {k}: {fnames_sel[k]}, label={int(y_sel[k])}"],
        )
        embedded.append(hdr_path)

    return ExportedTestVectors(
        vectors_dir=vectors_dir,
        images_root=images_root,
        manifest_path=manifest_path,
        golden_x94_f64_path=golden_f64_path,
        golden_x94_q15_path=golden_q15_path,
        indices=sel,
        file_names=fnames_sel,
        labels=y_sel,
        images_u8=u8_sel,
        images_f32=X_f32_sel,
        x94_f64=x94_f64,
        x94_q15=x94_q15,
        embedded_headers=tuple(embedded),
    )
