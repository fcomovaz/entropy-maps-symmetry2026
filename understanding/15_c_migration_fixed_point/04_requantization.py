"""Lab 15.4: Fixed-point convolution requantization, output shift, and accumulator headroom.

Manual execution (PowerShell from 'code/' directory):
    .\\.venv\\Scripts\\python.exe understanding/15_c_migration_fixed_point/04_requantization.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging import configure_logging, logger


def choose_frac_bits(max_abs: float, storage_bits: int = 16, headroom_bits: int = 0) -> int:
    """Select power-of-two fractional bits `n` for signed `storage_bits` without overflow."""
    int_bits_needed = 1 if max_abs <= 1e-12 else max(0, math.ceil(math.log2(max_abs))) + 1
    if max_abs > 0.0 and abs(max_abs - (1 << (int_bits_needed - 1))) < 1e-12:
        int_bits_needed += 1
    frac = storage_bits - 1 - int_bits_needed - headroom_bits
    return max(0, min(storage_bits - 1, frac))


def quantize_pow2(arr: np.ndarray, frac_bits: int, bits: int = 16) -> np.ndarray:
    """Quantize float array to integer with power-of-two scale 2^frac_bits."""
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    scaled = np.floor(np.asarray(arr, dtype=np.float64) * float(1 << frac_bits) + 0.5).astype(np.int64)
    return np.clip(scaled, qmin, qmax)


def round_shift_right_i64(arr: np.ndarray, shift: int) -> np.ndarray:
    """Bit-exact half-up arithmetic right shift `(x + 2^(s-1)) >> s` for int64 arrays."""
    x = np.asarray(arr, dtype=np.int64)
    if shift == 0:
        return x
    if shift < 0:
        return x << (-shift)
    return (x + (1 << (shift - 1))) >> shift


def conv2d_q15_reference(
    x_q: np.ndarray,
    w_q: np.ndarray,
    b_q: np.ndarray,
    shift_out: int,
    y_bits: int = 16,
    stride: int = 1,
    padding: int = 1,
) -> tuple[np.ndarray, int]:
    """Run single Conv2d layer using int64 accumulator and power-of-two output shift."""
    c_in, h_in, w_in = x_q.shape
    c_out, _, k_h, k_w = w_q.shape
    h_out = (h_in + 2 * padding - k_h) // stride + 1
    w_out = (w_in + 2 * padding - k_w) // stride + 1

    x_pad = np.pad(x_q, ((0, 0), (padding, padding), (padding, padding)), mode="constant", constant_values=0)
    y_q = np.empty((c_out, h_out, w_out), dtype=np.int64)
    max_acc_abs = 0

    for oc in range(c_out):
        for oh in range(h_out):
            iy0 = oh * stride
            for ow in range(w_out):
                ix0 = ow * stride
                patch = x_pad[:, iy0 : iy0 + k_h, ix0 : ix0 + k_w]
                acc = int(b_q[oc]) + int(np.sum(patch.astype(np.int64) * w_q[oc].astype(np.int64)))
                if abs(acc) > max_acc_abs:
                    max_acc_abs = abs(acc)
                shifted = int(round_shift_right_i64(np.array([acc]), shift_out)[0])
                qmin = -(1 << (y_bits - 1))
                qmax = (1 << (y_bits - 1)) - 1
                y_q[oc, oh, ow] = max(qmin, min(qmax, shifted))
    return y_q, max_acc_abs


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for Lab 15.4."""
    parser = argparse.ArgumentParser(description="Study Conv2d fixed-point requantization and accumulator headroom.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/understanding/c_migration_fixed_point.toml"),
        help="Path to Lab 15 TOML config relative to code/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Sweep fan-in Cin*k^2 to verify int64 headroom and q15 error vs float32."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        cfg_path = (PROJECT_ROOT / args.config).resolve()
        with cfg_path.open("rb") as f:
            cfg = tomllib.load(f)
        out_dir = (PROJECT_ROOT / cfg["experiment"]["output_dir"]).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        dpi = int(cfg["experiment"].get("dpi", 160))
        seed = int(cfg["experiment"]["seed"])

        rng = np.random.default_rng(seed)
        fan_ins = [9, 72, 144, 288, 576]
        rows = []
        for c_in in [1, 8, 16, 32, 64]:
            fan = c_in * 9
            x_f = rng.normal(0.0, 0.5, size=(c_in, 16, 16)).astype(np.float32)
            w_f = rng.normal(0.0, 0.25 / math.sqrt(c_in), size=(8, c_in, 3, 3)).astype(np.float32)
            b_f = rng.normal(0.0, 0.05, size=(8,)).astype(np.float32)

            with torch.no_grad():
                y_ref = torch.nn.functional.conv2d(
                    torch.from_numpy(x_f).unsqueeze(0),
                    torch.from_numpy(w_f),
                    torch.from_numpy(b_f),
                    stride=1,
                    padding=1,
                ).squeeze(0).numpy()

            x_frac = choose_frac_bits(float(np.max(np.abs(x_f))), 16)
            w_frac = choose_frac_bits(float(np.max(np.abs(w_f))), 16)
            y_frac = choose_frac_bits(float(np.max(np.abs(y_ref))), 16, headroom_bits=1)

            x_q = quantize_pow2(x_f, x_frac, 16)
            w_q = quantize_pow2(w_f, w_frac, 16)
            b_q = quantize_pow2(b_f, x_frac + w_frac, 64)
            shift_out = x_frac + w_frac - y_frac

            y_q, max_acc = conv2d_q15_reference(x_q, w_q, b_q, shift_out=shift_out)
            y_deq = y_q.astype(np.float64) / float(1 << y_frac)
            abs_err = np.abs(y_deq - y_ref)
            bits_used = math.ceil(math.log2(max_acc + 1)) if max_acc > 0 else 1

            rows.append(
                {
                    "fan_in": fan,
                    "x_frac": x_frac,
                    "w_frac": w_frac,
                    "y_frac": y_frac,
                    "shift_out": shift_out,
                    "max_acc_bits": bits_used,
                    "max_abs_err": float(np.max(abs_err)),
                    "mean_abs_err": float(np.mean(abs_err)),
                }
            )

        with (out_dir / "04_requantization.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        fans = [r["fan_in"] for r in rows]
        axes[0].plot(fans, [r["max_acc_bits"] for r in rows], "o-", label="Observed acc bits")
        axes[0].axhline(31, color="r", linestyle="--", label="int32 signed limit (31 bits)")
        axes[0].set_title("Accumulator Bit-Width vs Fan-In (Cin * k^2)")
        axes[0].set_xlabel("Cin * k^2")
        axes[0].set_ylabel("Bits required (magnitude)")
        axes[0].legend()

        axes[1].plot(fans, [r["mean_abs_err"] for r in rows], "s-", label="Mean |delta|")
        axes[1].plot(fans, [r["max_abs_err"] for r in rows], "^-", label="Max |delta|")
        axes[1].set_title("Conv2d Power-of-Two Quantization Error")
        axes[1].set_xlabel("Cin * k^2")
        axes[1].set_ylabel("Absolute error")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(out_dir / "04_requantization.png", dpi=dpi)
        plt.close(fig)

        logger.info("Lab 15.4 complete: outputs -> %s", out_dir)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
