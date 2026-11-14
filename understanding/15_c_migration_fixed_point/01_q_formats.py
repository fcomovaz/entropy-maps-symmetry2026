"""Lab 15.1: Qm.n fixed-point formats, cimp_q15_mul_round, and saturation vs wrap-around.

Manual execution (PowerShell from 'code/' directory):
    .\\.venv\\Scripts\\python.exe understanding/15_c_migration_fixed_point/01_q_formats.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging import configure_logging, logger


def sat_i16(x: int) -> int:
    """Saturate signed integer to int16_t [-32768, 32767]."""
    if x > 32767:
        return 32767
    if x < -32768:
        return -32768
    return int(x)


def wrap_i16(x: int) -> int:
    """Wrap signed integer with two's complement int16_t overflow (for contrast)."""
    v = int(x) & 0xFFFF
    return v - 0x10000 if (v & 0x8000) else v


def q15_mul_round(a: int, b: int) -> int:
    """Bit-exact Python mirror of the Q0.15 multiply-round-saturate C kernel.

    Mirrors:
        int32_t prod = ((int32_t)a * b) + (1 << 14);
        if (prod >= 0x40000000) return 32767;
        return (q15_t)(prod >> 15);
    """
    prod = int(a) * int(b) + (1 << 14)
    if prod >= 0x40000000:
        return 32767
    return int(prod >> 15)


def quantize_array(x: np.ndarray, frac_bits: int, total_bits: int = 16) -> tuple[np.ndarray, int]:
    """Quantize floating-point values into signed Qm.n integers with half-up rounding and saturation."""
    scale = float(1 << frac_bits)
    qmin = -(1 << (total_bits - 1))
    qmax = (1 << (total_bits - 1)) - 1
    scaled = np.floor(np.asarray(x, dtype=np.float64) * scale + 0.5).astype(np.int64)
    saturated = int(np.sum((scaled < qmin) | (scaled > qmax)))
    return np.clip(scaled, qmin, qmax).astype(np.int64), saturated


def dequantize_array(q: np.ndarray, frac_bits: int) -> np.ndarray:
    """Convert signed fixed-point integer array back to float64."""
    return np.asarray(q, dtype=np.float64) / float(1 << frac_bits)


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for Lab 15.1."""
    parser = argparse.ArgumentParser(description="Study Qm.n fixed-point representation and Q15 multiplication.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/understanding/c_migration_fixed_point.toml"),
        help="Path to Lab 15 TOML config relative to code/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run Lab 15.1 Q-format experiments and generate CSV + diagnostic plot."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        cfg_path = (PROJECT_ROOT / args.config).resolve()
        with cfg_path.open("rb") as f:
            cfg = tomllib.load(f)
        out_dir = (PROJECT_ROOT / cfg["experiment"]["output_dir"]).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = int(cfg["experiment"]["seed"])
        dpi = int(cfg["experiment"].get("dpi", 160))

        rng = np.random.default_rng(seed)
        grid = rng.uniform(-0.9999, 0.9999, size=4096)
        q15_vals, n_sat = quantize_array(grid, frac_bits=15, total_bits=16)
        deq = dequantize_array(q15_vals, frac_bits=15)
        abs_err = np.abs(grid - deq)

        # Saturation vs wrap-around demonstration including -32768 * -32768 and sum overflow
        edge_cases = [
            ("neg_one_sq", -32768, -32768, q15_mul_round(-32768, -32768), wrap_i16((-32768 * -32768 + (1 << 14)) >> 15)),
            ("neg_half_neg_half", -16384, -16384, q15_mul_round(-16384, -16384), wrap_i16((-16384 * -16384 + (1 << 14)) >> 15)),
            ("neg_tiny_pos", -1, 16384, q15_mul_round(-1, 16384), wrap_i16((-1 * 16384 + (1 << 14)) >> 15)),
            ("pos_max_sq", 32767, 32767, q15_mul_round(32767, 32767), wrap_i16((32767 * 32767 + (1 << 14)) >> 15)),
        ]

        # Test Q0.15 * Q0.15 multiplication accuracy across 4096 random pairs
        a_f = rng.uniform(-1.0, 0.9999, size=4096)
        b_f = rng.uniform(-1.0, 0.9999, size=4096)
        a_q, _ = quantize_array(a_f, 15, 16)
        b_q, _ = quantize_array(b_f, 15, 16)
        mul_q = np.array([q15_mul_round(int(ai), int(bi)) for ai, bi in zip(a_q, b_q)], dtype=np.int64)
        mul_err = np.abs(dequantize_array(mul_q, 15) - (a_f * b_f))

        # Check saturation vs wrap-around on accumulator sum beyond Q0.15
        acc_f = np.linspace(0.5, 1.5, 200)
        acc_q_raw = np.floor(acc_f * 32768.0 + 0.5).astype(np.int64)
        acc_sat = np.array([sat_i16(int(v)) for v in acc_q_raw]) / 32768.0
        acc_wrap = np.array([wrap_i16(int(v)) for v in acc_q_raw]) / 32768.0

        csv_path = out_dir / "01_q_formats_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["q15_quant_max_abs_err", f"{float(np.max(abs_err)):.8e}"])
            writer.writerow(["q15_quant_mean_abs_err", f"{float(np.mean(abs_err)):.8e}"])
            writer.writerow(["q15_mul_max_abs_err", f"{float(np.max(mul_err)):.8e}"])
            writer.writerow(["q15_mul_mean_abs_err", f"{float(np.mean(mul_err)):.8e}"])
            writer.writerow(["q15_grid_saturated_count", n_sat])
            for label, a_i, b_i, sat_r, wrap_r in edge_cases:
                writer.writerow([f"edge_{label}_sat_vs_wrap", f"a={a_i},b={b_i},sat={sat_r},wrap={wrap_r}"])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].hist(abs_err, bins=32, alpha=0.75, label="Direct Q0.15 quant")
        axes[0].hist(mul_err, bins=32, alpha=0.65, label="Q0.15 * Q0.15 mul")
        axes[0].set_title("Q0.15 Quantization & Multiply Error")
        axes[0].set_xlabel("|delta|")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        axes[1].plot(acc_f, acc_f, "k--", label="Ideal float64")
        axes[1].plot(acc_f, acc_sat, label="Saturating int16")
        axes[1].plot(acc_f, acc_wrap, alpha=0.8, label="Wrap-around int16")
        axes[1].set_title("Saturation vs Wrap-around (> 1.0)")
        axes[1].set_xlabel("True real value")
        axes[1].set_ylabel("Represented value")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(out_dir / "01_q_formats.png", dpi=dpi)
        plt.close(fig)

        logger.info(
            "Lab 15.1 complete: q15 max_err=%.3e, mul max_err=%.3e, outputs -> %s",
            float(np.max(abs_err)),
            float(np.max(mul_err)),
            out_dir,
        )
        return 0
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
