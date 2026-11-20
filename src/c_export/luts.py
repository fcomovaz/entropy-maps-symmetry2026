"""Deterministic C99 LUT and configuration header generators.

Generates:
    - cimp_config.h       : dataset/grid/bin/scale constants, asymmetry schema, arena bounds
    - lut_bin_quant.h     : 256-entry uint8 pixel -> bin LUT (reusing src.features.entropy_maps)
    - lut_shannon.h       : fixed-point Shannon LUT tables in configured mode
    - lut_cnn_input.h     : 256-entry CNN pixel normalization LUT in f32 and q15
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from src.features.entropy_maps import build_quantization_lut

from .c_writer import format_c_array, format_c_float, sha256_file, write_header
from .config import CImpConfig
from .fixed_point import quantize


@dataclass(frozen=True, slots=True)
class ShannonLutBundle:
    """In-memory Shannon LUT tables and metadata shared between C emission and Python mirror."""

    mode: str
    feature_frac_bits: int
    nlogn_frac_bits: int
    mantissa_bits: int
    nlogn_table: np.ndarray        # uint32[max_pixels + 1]
    term_table: np.ndarray         # int16[sum(N_l + 1)]
    term_offsets: np.ndarray       # int32[n_levels]
    mantissa_table: np.ndarray     # uint16[2^k + 1]
    level_npix: np.ndarray         # int32[n_levels]
    level_log2_n_q15: np.ndarray   # uint32[n_levels]
    byte_size: int


def build_bin_quant_lut(bins: int) -> np.ndarray:
    """Build 256-element uint8 pixel -> bin LUT via formal src helper."""
    return build_quantization_lut(bins)


def build_shannon_lut(
    image_size: int,
    grid_levels: Sequence[int],
    bins: int,
    mode: str,
    feature_frac_bits: int = 15,
    mantissa_bits: int = 8,
) -> ShannonLutBundle:
    """Build Shannon entropy lookup tables for the selected mode."""
    levels_asc = tuple(sorted(int(g) for g in grid_levels))
    level_npix = np.array([(image_size // g) ** 2 for g in levels_asc], dtype=np.int32)
    level_log2_n_q15 = np.array(
        [int(math.floor(math.log2(int(npix)) * 32768.0 + 0.5)) for npix in level_npix],
        dtype=np.uint32,
    )

    max_npix = int(np.max(level_npix))
    max_nlogn = float(max_npix) * math.log2(max(2, max_npix))
    nlogn_int_bits = max(1, math.ceil(math.log2(max_nlogn + 1e-9)))
    nlogn_frac_bits = max(8, min(20, 31 - nlogn_int_bits))

    c_arr = np.arange(max_npix + 1, dtype=np.float64)
    nlogn_table = np.zeros(max_npix + 1, dtype=np.uint32)
    if max_npix >= 1:
        scale_nlogn = math.ldexp(1.0, nlogn_frac_bits)
        vals = np.floor(c_arr[1:] * np.log2(c_arr[1:]) * scale_nlogn + 0.5)
        nlogn_table[1:] = vals.astype(np.uint32)

    log2_b = math.log2(bins)
    scale_feat = math.ldexp(1.0, feature_frac_bits)
    qmax_feat = min(32767, (1 << feature_frac_bits) - 1)

    term_chunks: list[np.ndarray] = []
    offsets_list: list[int] = []
    cur_off = 0
    for npix in level_npix:
        offsets_list.append(cur_off)
        n_int = int(npix)
        chunk = np.zeros(n_int + 1, dtype=np.int16)
        p = np.arange(1, n_int + 1, dtype=np.float64) / float(n_int)
        raw = np.floor((-p * np.log2(p) / log2_b) * scale_feat + 0.5).astype(np.int64)
        chunk[1:] = np.clip(raw, 0, qmax_feat).astype(np.int16)
        term_chunks.append(chunk)
        cur_off += n_int + 1
    term_table = np.concatenate(term_chunks).astype(np.int16)
    term_offsets = np.asarray(offsets_list, dtype=np.int32)

    m_size = 1 << mantissa_bits
    mant_vals = [
        int(math.floor(math.log2(1.0 + float(m) / float(m_size)) * 32768.0 + 0.5))
        for m in range(m_size + 1)
    ]
    mantissa_table = np.asarray(mant_vals, dtype=np.uint16)

    if mode == "nlogn_full":
        byte_size = int(nlogn_table.nbytes)
    elif mode == "term_per_level":
        byte_size = int(term_table.nbytes + term_offsets.nbytes)
    elif mode == "log2_mantissa":
        byte_size = int(mantissa_table.nbytes)
    else:
        raise ValueError(f"Unsupported Shannon LUT mode '{mode}'")

    return ShannonLutBundle(
        mode=mode,
        feature_frac_bits=feature_frac_bits,
        nlogn_frac_bits=nlogn_frac_bits,
        mantissa_bits=mantissa_bits,
        nlogn_table=nlogn_table,
        term_table=term_table,
        term_offsets=term_offsets,
        mantissa_table=mantissa_table,
        level_npix=level_npix,
        level_log2_n_q15=level_log2_n_q15,
        byte_size=byte_size,
    )


def build_cnn_input_lut(
    mean: float,
    std: float,
    frac_bits: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Precompute 256-entry float32 and Q15 normalization LUTs for uint8 grayscale pixels."""
    if std <= 0.0:
        raise ValueError(f"normalize_std must be > 0, got {std}")
    p = np.arange(256, dtype=np.float64) / 255.0
    norm = (p - mean) / std
    f32_lut = np.ascontiguousarray(norm, dtype=np.float32)
    q_arr, n_sat = quantize(norm, frac_bits=frac_bits, bits=16)
    return f32_lut, q_arr.astype(np.int16), n_sat


def _mode_code(mode: str) -> int:
    mapping = {"nlogn_full": 0, "term_per_level": 1, "log2_mantissa": 2}
    if mode not in mapping:
        raise ValueError(f"Unknown Shannon mode '{mode}'")
    return mapping[mode]


def _kind_code(kind: str) -> int:
    mapping = {"lr": 0, "tb": 1, "rot180": 2}
    if kind not in mapping:
        raise ValueError(f"Unknown asymmetry kind '{kind}'")
    return mapping[kind]


def emit_config_and_lut_headers(cfg: CImpConfig) -> dict[str, dict[str, object]]:
    """Generate cimp_config.h, lut_bin_quant.h, lut_shannon.h, and lut_cnn_input.h."""
    out_dir = cfg.generated_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = cfg.descriptor_spec
    provenance = [
        f"Target tag: {cfg.effective_target_tag}",
        f"Config: {cfg.config_path.name} (sha256={sha256_file(cfg.config_path)[:16]})",
    ]

    # 1. cimp_config.h
    g_max = max(spec.grid_levels)
    levels_asc = sorted(spec.grid_levels)
    coarse_cells = sum(g * g for g in levels_asc if g != g_max)
    max_cell_pix = (spec.image_size // min(levels_asc)) ** 2
    count_ctype = "uint16_t" if max_cell_pix <= 65535 else "uint32_t"
    log2_b_q15 = int(math.floor(math.log2(spec.bins) * 32768.0 + 0.5))

    # Precompute reciprocal Q15 multipliers for asymmetry 1/(g*g) if non-power-of-two,
    # and exact shift when g*g is power-of-two.
    asym_div_shift = []
    asym_inv_q31 = []
    for g in spec.asym_scales:
        denom = g * g
        is_pow2 = (denom & (denom - 1)) == 0
        asym_div_shift.append(denom.bit_length() - 1 if is_pow2 else -1)
        asym_inv_q31.append(int(round((1 << 31) / float(denom))))

    bundle = build_shannon_lut(
        image_size=spec.image_size,
        grid_levels=levels_asc,
        bins=spec.bins,
        mode=cfg.descriptor.shannon_lut_mode,
        feature_frac_bits=cfg.numeric.feature_frac_bits,
        mantissa_bits=cfg.descriptor.log2_mantissa_bits,
    )

    cfg_body_lines = [
        f'#define CIMP_TARGET_TAG "{cfg.effective_target_tag}"',
        f'#define CIMP_DESCRIPTOR_SCHEMA "{spec.schema}"',
        f"#define CIMP_IMAGE_SIZE {spec.image_size}",
        f"#define CIMP_CNN_INPUT_SIZE {cfg.cnn_cfg.data.image_size}",
        f"#define CIMP_N_LEVELS {len(levels_asc)}",
        f"#define CIMP_BINS {spec.bins}",
        f"#define CIMP_FINEST_GRID {g_max}",
        f"#define CIMP_FINEST_CELLS {g_max * g_max}",
        f"#define CIMP_COARSE_CELLS {max(1, coarse_cells)}",
        f"#define CIMP_CELL_SIDE {spec.image_size // g_max}",
        f"#define CIMP_N_X85 {spec.n_x85}",
        f"#define CIMP_N_ASYM_SCALES {len(spec.asym_scales)}",
        f"#define CIMP_N_ASYM_KINDS {len(spec.asym_kinds)}",
        f"#define CIMP_N_X9 {spec.n_x9}",
        f"#define CIMP_N_X94 {spec.n_x94}",
        f"#define CIMP_FEATURE_FRAC_BITS {cfg.numeric.feature_frac_bits}",
        f"#define CIMP_FEATURE_Q_MAX {min(32767, (1 << cfg.numeric.feature_frac_bits) - 1)}",
        f"#define CIMP_LOG2_BINS_Q15 {log2_b_q15}",
        f"#define CIMP_LOG2_BINS_F32 {format_c_float(math.log2(spec.bins))}",
        f"#define CIMP_SHANNON_MODE_NLOGN_FULL 0",
        f"#define CIMP_SHANNON_MODE_TERM_PER_LEVEL 1",
        f"#define CIMP_SHANNON_MODE_LOG2_MANTISSA 2",
        f"#define CIMP_SHANNON_LUT_MODE {_mode_code(cfg.descriptor.shannon_lut_mode)}",
        f"#define CIMP_LOG2_MANTISSA_BITS {cfg.descriptor.log2_mantissa_bits}",
        f"#define CIMP_NLOGN_FRAC_BITS {bundle.nlogn_frac_bits}",
        f"#define CIMP_CNN_INPUT_FRAC_BITS {cfg.cnn.input_frac_bits}",
        f"#define CIMP_FLASH_BUDGET_BYTES {cfg.cnn.flash_budget_bytes}",
        f"#define CIMP_SRAM_BUDGET_BYTES {cfg.cnn.sram_budget_bytes}",
        f"#define CIMP_PSRAM_BUDGET_BYTES {cfg.cnn.psram_budget_bytes}",
        f"typedef {count_ctype} cimp_hist_count_t;",
        format_c_array("CIMP_LEVELS", levels_asc, "int32_t", per_line=8, rom=True),
        format_c_array("CIMP_LEVEL_NPIX", bundle.level_npix, "int32_t", per_line=8, rom=True),
        format_c_array("CIMP_LEVEL_LOG2_N_Q15", bundle.level_log2_n_q15, "uint32_t", per_line=8, rom=True),
        format_c_array("CIMP_ASYM_SCALES", spec.asym_scales, "int32_t", per_line=8, rom=True),
        format_c_array("CIMP_ASYM_KINDS", [_kind_code(k) for k in spec.asym_kinds], "int32_t", per_line=8, rom=True),
        format_c_array("CIMP_ASYM_DIV_SHIFT", asym_div_shift, "int32_t", per_line=8, rom=True),
        format_c_array("CIMP_ASYM_INV_Q31", asym_inv_q31, "int32_t", per_line=8, rom=True),
    ]
    sha_cfg = write_header(
        out_dir / "cimp_config.h",
        "CIMP_GENERATED_CONFIG_H",
        ["<stdint.h>", '"cimp_fixed_point.h"'],
        "\n\n".join(cfg_body_lines),
        provenance,
    )

    # 2. lut_bin_quant.h
    qlut = build_bin_quant_lut(spec.bins)
    sha_qlut = write_header(
        out_dir / "lut_bin_quant.h",
        "CIMP_LUT_BIN_QUANT_H",
        ["<stdint.h>", '"cimp_fixed_point.h"'],
        format_c_array("CIMP_LUT_BIN_QUANT", qlut, "uint8_t", per_line=16, rom=True),
        provenance + [f"Quantization bins: {spec.bins}"],
    )

    # 3. lut_shannon.h (emits active mode tables to keep binary lean; C runtime dispatches on CIMP_SHANNON_LUT_MODE)
    shannon_parts = [
        f"/* Active Shannon LUT mode: {bundle.mode} ({bundle.byte_size} bytes) */",
        format_c_array("CIMP_LUT_SHANNON_TERM_OFFSETS", bundle.term_offsets, "int32_t", per_line=8, rom=True),
    ]
    if bundle.mode == "nlogn_full":
        shannon_parts.append(format_c_array("CIMP_LUT_NLOGN", bundle.nlogn_table, "uint32_t", per_line=10, rom=True))
    elif bundle.mode == "term_per_level":
        shannon_parts.append(format_c_array("CIMP_LUT_SHANNON_TERM", bundle.term_table, "int16_t", per_line=12, rom=True))
    else:
        shannon_parts.append(format_c_array("CIMP_LUT_LOG2_MANTISSA", bundle.mantissa_table, "uint16_t", per_line=12, rom=True))

    sha_shannon = write_header(
        out_dir / "lut_shannon.h",
        "CIMP_LUT_SHANNON_H",
        ["<stdint.h>", '"cimp_fixed_point.h"', '"cimp_config.h"'],
        "\n\n".join(shannon_parts),
        provenance + [f"Mode: {bundle.mode}, bytes={bundle.byte_size}"],
    )

    # 4. lut_cnn_input.h
    cnn_f32_lut, cnn_q15_lut, n_sat = build_cnn_input_lut(
        cfg.cnn_cfg.data.normalize_mean,
        cfg.cnn_cfg.data.normalize_std,
        cfg.cnn.input_frac_bits,
    )
    cnn_lut_body = "\n\n".join(
        [
            f"#define CIMP_CNN_NORM_MEAN_F32 {format_c_float(cfg.cnn_cfg.data.normalize_mean)}",
            f"#define CIMP_CNN_NORM_STD_F32 {format_c_float(cfg.cnn_cfg.data.normalize_std)}",
            f"#define CIMP_CNN_INPUT_SATURATED_ENTRIES {n_sat}",
            format_c_array(
                "CIMP_LUT_CNN_INPUT_F32",
                cnn_f32_lut,
                "float",
                per_line=8,
                rom=True,
                guard_macro="CIMP_ENABLE_F32",
            ),
            format_c_array("CIMP_LUT_CNN_INPUT_Q15", cnn_q15_lut, "int16_t", per_line=12, rom=True),
        ]
    )
    sha_cnn_lut = write_header(
        out_dir / "lut_cnn_input.h",
        "CIMP_LUT_CNN_INPUT_H",
        ["<stdint.h>", '"cimp_fixed_point.h"'],
        cnn_lut_body,
        provenance + [f"Normalization: mean={cfg.cnn_cfg.data.normalize_mean}, std={cfg.cnn_cfg.data.normalize_std}"],
    )

    return {
        "cimp_config.h": {"sha256": sha_cfg, "bytes": (out_dir / "cimp_config.h").stat().st_size},
        "lut_bin_quant.h": {"sha256": sha_qlut, "bytes": int(qlut.nbytes)},
        "lut_shannon.h": {"sha256": sha_shannon, "mode": bundle.mode, "bytes": bundle.byte_size},
        "lut_cnn_input.h": {"sha256": sha_cnn_lut, "saturated_entries": n_sat, "bytes": int(cnn_f32_lut.nbytes + cnn_q15_lut.nbytes)},
    }
