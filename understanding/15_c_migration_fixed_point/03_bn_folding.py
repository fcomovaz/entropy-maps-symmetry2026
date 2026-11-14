"""Lab 15.3: BatchNorm2d folding into Conv2d and numerical parity on trained CNN checkpoints.

Manual execution (PowerShell from 'code/' directory):
    .\\.venv\\Scripts\\python.exe understanding/15_c_migration_fixed_point/03_bn_folding.py
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
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging import configure_logging, logger
from src.models.cnn.architectures.registry import create_model


def fold_conv_bn_weights(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytically fold BatchNorm2d (eval mode) parameters into preceding Conv2d weight & bias."""
    with torch.no_grad():
        gamma = bn.weight if bn.weight is not None else torch.ones(conv.out_channels, dtype=conv.weight.dtype)
        beta = bn.bias if bn.bias is not None else torch.zeros(conv.out_channels, dtype=conv.weight.dtype)
        mean = bn.running_mean
        var = bn.running_var
        inv_std = gamma / torch.sqrt(var + bn.eps)

        w_scale = inv_std.view(-1, 1, 1, 1)
        w_fold = conv.weight.detach() * w_scale
        b_orig = conv.bias.detach() if conv.bias is not None else torch.zeros(conv.out_channels, dtype=conv.weight.dtype)
        b_fold = beta + (b_orig - mean) * inv_std
        return w_fold.clone(), b_fold.clone()


def verify_conv_bn_fold_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d, x: torch.Tensor) -> float:
    """Return max absolute difference between conv+bn(eval) and single folded Conv2d."""
    conv.eval()
    bn.eval()
    w_fold, b_fold = fold_conv_bn_weights(conv, bn)
    folded = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )
    with torch.no_grad():
        folded.weight.copy_(w_fold)
        folded.bias.copy_(b_fold)
        y_ref = bn(conv(x))
        y_fold = folded(x)
        return float(torch.max(torch.abs(y_ref - y_fold)).item())


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for Lab 15.3."""
    parser = argparse.ArgumentParser(description="Verify analytical Conv2d+BatchNorm2d folding on trained CNN.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/understanding/c_migration_fixed_point.toml"),
        help="Path to Lab 15 TOML config relative to code/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Verify Conv2d + BatchNorm2d folding across all layers of the sample CNN."""
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

        ckpt_path = (PROJECT_ROOT / cfg["cnn_sample"]["checkpoint_path"]).resolve()
        arch = str(cfg["cnn_sample"]["architecture"])
        in_size = int(cfg["cnn_sample"]["input_size"])

        model = create_model(arch, num_classes=2)
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)
        model.eval()

        torch.manual_seed(seed)
        x_cur = torch.randn(4, 1, in_size, in_size)

        # Inspect stem and depthwise-separable blocks of ultra_tiny
        layer_diffs: list[tuple[str, float, float, float]] = []

        def _record_pair(name: str, conv: nn.Conv2d, bn: nn.BatchNorm2d, inp: torch.Tensor) -> torch.Tensor:
            diff = verify_conv_bn_fold_pair(conv, bn, inp)
            w_fold, _ = fold_conv_bn_weights(conv, bn)
            layer_diffs.append(
                (
                    name,
                    diff,
                    float(torch.max(torch.abs(conv.weight)).item()),
                    float(torch.max(torch.abs(w_fold)).item()),
                )
            )
            with torch.no_grad():
                return bn(conv(inp))

        h = _record_pair("stem.conv+bn", model.stem[0], model.stem[1], x_cur)
        h = torch.relu(h)
        h = _record_pair("block1.depthwise+bn", model.block1.depthwise, model.block1.depthwise_bn, h)
        h = torch.relu(h)
        h = _record_pair("block1.pointwise+bn", model.block1.pointwise, model.block1.pointwise_bn, h)
        h = torch.relu(h)
        h = _record_pair("block2.depthwise+bn", model.block2.depthwise, model.block2.depthwise_bn, h)
        h = torch.relu(h)
        h = _record_pair("block2.pointwise+bn", model.block2.pointwise, model.block2.pointwise_bn, h)

        with (out_dir / "03_bn_folding.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["layer_pair", "max_abs_diff", "raw_w_max_abs", "folded_w_max_abs"])
            for name, d, rw, fw in layer_diffs:
                writer.writerow([name, f"{d:.6e}", f"{rw:.6f}", f"{fw:.6f}"])

        fig, ax = plt.subplots(figsize=(8, 4.0))
        names = [row[0] for row in layer_diffs]
        diffs = [max(row[1], 1e-12) for row in layer_diffs]
        ax.bar(names, diffs)
        ax.set_yscale("log")
        ax.set_ylabel("Max |Conv+BN - Folded Conv|")
        ax.set_title(f"BatchNorm2d Folding Error ({arch})")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / "03_bn_folding.png", dpi=dpi)
        plt.close(fig)

        logger.info("Lab 15.3 complete: max diff across pairs = %.3e", max(r[1] for r in layer_diffs))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
