"""Post-run verification and visual validation for KolektorSDD2 preprocessing.

Manual execution commands:
    # Verify standard preprocessed dataset:
    .venv\\Scripts\\python.exe understanding/10_verify_kolektorsdd2/verify.py

    # Verify inverted dataset:
    .venv\\Scripts\\python.exe understanding/10_verify_kolektorsdd2/verify.py --inverted
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.kolektorsdd2 import discover_pairs, load_config


def compute_directory_size_mb(directory: Path, extension: str) -> float:
    """Compute exact total disk size in MB for files matching extension."""
    if not directory.exists():
        return 0.0
    ext = extension.lower()
    total_bytes = sum(
        f.stat().st_size
        for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() == ext
    )
    return total_bytes / (1024 * 1024)


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for verification utility."""
    parser = argparse.ArgumentParser(
        prog="verify.py",
        description="Verify preprocessed KolektorSDD2 dataset integrity and visualize samples.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocessing/kolektorsdd2.toml"),
        help="Path to preprocessing configuration TOML file.",
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        default=False,
        help="Verify the inverted preprocessed dataset (KolektorSDD2-inverted).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample visualization.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Verify preprocessing integrity, class distribution, and generate visual comparison."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config, PROJECT_ROOT)
    target_output = cfg.output_root_inverted if args.inverted else cfg.output_root

    print("=" * 70)
    print(f"KolektorSDD2 Preprocessing Post-Run Verification (Inverted={args.inverted})")
    print("=" * 70)
    print(f"Raw directory:    {cfg.input_root}")
    print(f"Target directory: {target_output}")

    if not target_output.exists():
        print(f"ERROR: Target directory does not exist: {target_output}")
        cmd_flag = " --inverted" if args.inverted else ""
        print("Please execute preprocessing first via:")
        print(f"    .venv\\Scripts\\python.exe -m experiments.preprocess_kolektorsdd2{cmd_flag}")
        return 1

    # Measure disk usage
    raw_mb = compute_directory_size_mb(cfg.input_root, cfg.image_extension)
    proc_mb = compute_directory_size_mb(target_output, cfg.image_extension)
    print(f"Disk footprint:   Raw: {raw_mb:.2f} MB | Processed: {proc_mb:.2f} MB")

    # Count files per split and category
    counts: dict[str, dict[str, int]] = {}
    invalid_modes = 0
    invalid_sizes = 0
    mask_files_found = 0
    total_processed = 0

    for split in cfg.splits:
        counts[split] = {}
        for category in (cfg.good_folder, cfg.defect_folder):
            cat_dir = target_output / split / category
            if not cat_dir.is_dir():
                counts[split][category] = 0
                continue
            files = [
                f for f in sorted(cat_dir.iterdir())
                if f.is_file() and f.suffix.lower() == cfg.image_extension.lower()
            ]
            counts[split][category] = len(files)
            total_processed += len(files)

            for f in files:
                if f.name.endswith(f"{cfg.mask_suffix}{cfg.image_extension}"):
                    mask_files_found += 1
                with Image.open(f) as img:
                    if img.mode != "L":
                        invalid_modes += 1
                    if img.size != (cfg.target_size, cfg.target_size):
                        invalid_sizes += 1

    print("\n--- Category Breakdown ---")
    print(f"{'Split':<10} | {'Good':<10} | {'Defect':<10} | {'Total':<10} | {'Expected (Good/Defect)':<25}")
    print("-" * 70)

    expected = {
        "train": (2085, 246),
        "test": (894, 110),
    }

    for split in cfg.splits:
        good_cnt = counts[split].get(cfg.good_folder, 0)
        defect_cnt = counts[split].get(cfg.defect_folder, 0)
        tot_cnt = good_cnt + defect_cnt
        exp_good, exp_defect = expected.get(split, (-1, -1))
        exp_str = f"{exp_good} / {exp_defect}"
        match = "MATCH" if (good_cnt == exp_good and defect_cnt == exp_defect) else "MISMATCH"
        print(f"{split:<10} | {good_cnt:<10} | {defect_cnt:<10} | {tot_cnt:<10} | {exp_str:<15} [{match}]")

    print("-" * 70)
    print(f"Total processed images: {total_processed}")

    # Validation assertions
    has_errors = False
    if invalid_modes > 0:
        print(f"ERROR: {invalid_modes} images are not mode 'L'")
        has_errors = True
    else:
        print("PASSED: All images are mode 'L' (grayscale).")

    if invalid_sizes > 0:
        print(f"ERROR: {invalid_sizes} images do not match target size {cfg.target_size}x{cfg.target_size}")
        has_errors = True
    else:
        print(f"PASSED: All images match target size {cfg.target_size}x{cfg.target_size}.")

    if mask_files_found > 0:
        print(f"ERROR: Found {mask_files_found} mask files in processed directory")
        has_errors = True
    else:
        print("PASSED: No mask files exist in processed output directory.")

    if has_errors:
        return 1

    # Visual verification figure
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    suffix = "_inverted" if args.inverted else ""
    fig_path = output_dir / f"kolektorsdd2{suffix}_verification.png"

    rng = random.Random(args.seed)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(
        f"KolektorSDD2 Verification (Inverted={args.inverted})",
        fontsize=16,
        fontweight="bold",
    )

    row_targets = [
        ("train", cfg.good_folder),
        ("train", cfg.defect_folder),
        ("test", cfg.good_folder),
        ("test", cfg.defect_folder),
    ]

    for row_idx, (split, category) in enumerate(row_targets):
        cat_dir = target_output / split / category
        files = sorted(cat_dir.glob(f"*{cfg.image_extension}"))
        if not files:
            continue

        chosen_file = rng.choice(files)
        sample_id = chosen_file.stem
        raw_img_path = cfg.input_root / split / f"{sample_id}{cfg.image_extension}"
        raw_mask_path = cfg.input_root / split / f"{sample_id}{cfg.mask_suffix}{cfg.image_extension}"

        # 1. Raw image
        ax_raw = axes[row_idx, 0]
        with Image.open(raw_img_path) as img_raw:
            raw_arr = np.asarray(img_raw)
        ax_raw.imshow(raw_arr, cmap="gray" if raw_arr.ndim == 2 else None)
        ax_raw.set_title(f"Raw {split}/{category}\n{raw_img_path.name} ({raw_arr.shape})", fontsize=10)
        ax_raw.axis("off")

        # 2. Raw mask
        ax_mask = axes[row_idx, 1]
        with Image.open(raw_mask_path) as img_mask:
            mask_arr = np.asarray(img_mask)
        ax_mask.imshow(mask_arr, cmap="gray")
        ax_mask.set_title(f"Raw Mask\nmax={mask_arr.max()} ({mask_arr.shape})", fontsize=10)
        ax_mask.axis("off")

        # 3. Processed image
        ax_proc = axes[row_idx, 2]
        with Image.open(chosen_file) as img_proc:
            proc_arr = np.asarray(img_proc)
        ax_proc.imshow(proc_arr, cmap="gray", vmin=0, vmax=255)
        ax_proc.set_title(f"Processed 128x128\n{chosen_file.name}", fontsize=10)
        ax_proc.axis("off")

        # 4. Histograms
        ax_hist = axes[row_idx, 3]
        raw_gray = raw_arr if raw_arr.ndim == 2 else np.mean(raw_arr, axis=2)
        ax_hist.hist(raw_gray.ravel(), bins=32, range=(0, 256), alpha=0.5, label="Raw (gray)", color="blue", density=True)
        ax_hist.hist(proc_arr.ravel(), bins=32, range=(0, 256), alpha=0.5, label="Processed", color="green", density=True)
        ax_hist.set_title("Intensity Distribution", fontsize=10)
        ax_hist.legend(loc="upper right", fontsize=8)
        ax_hist.set_xlim(0, 256)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nVerification visual figure saved to: {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
