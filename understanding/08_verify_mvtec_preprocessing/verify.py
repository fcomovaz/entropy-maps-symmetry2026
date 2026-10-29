"""Post-run verification and visual validation for MVTec-AD preprocessing.

Manual execution commands:
    # Verify standard preprocessed dataset:
    .venv\\Scripts\\python.exe understanding/08_verify_mvtec_preprocessing/verify.py

    # Verify inverted dataset:
    .venv\\Scripts\\python.exe understanding/08_verify_mvtec_preprocessing/verify.py --inverted
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.letterbox import invert_intensity, letterbox_to_square
from src.preprocessing.mvtec_ad import is_mask, iter_images, load_config


def compute_directory_size_mb(directory: Path, extensions: tuple[str, ...]) -> float:
    """Compute exact total disk size in MB for files matching extensions."""
    if not directory.exists():
        return 0.0
    exts = set(extensions)
    total_bytes = sum(
        f.stat().st_size for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in exts
    )
    return total_bytes / (1024 * 1024)


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for verification utility."""
    parser = argparse.ArgumentParser(
        prog="verify.py",
        description="Verify preprocessed MVTec-AD dataset integrity and visualize samples.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocessing/mvtec_ad.toml"),
        help="Path to preprocessing configuration TOML file.",
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        default=False,
        help="Verify the inverted preprocessed dataset (mvtec-ad-inverted).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Verify preprocessing integrity, size reduction, and generate visual comparison."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config, PROJECT_ROOT)
    target_output = cfg.output_root_inverted if args.inverted else cfg.output_root

    print("=" * 60)
    print(f"MVTec-AD Preprocessing Post-Run Verification (Inverted={args.inverted})")
    print("=" * 60)
    print(f"Raw directory:          {cfg.input_root}")
    print(f"Target directory:       {target_output}")

    if not target_output.exists():
        print(f"ERROR: Target directory does not exist: {target_output}")
        cmd_flag = " --inverted" if args.inverted else ""
        print("Please execute preprocessing first via:")
        print(f"    .venv\\Scripts\\python.exe -m experiments.preprocess_mvtec_ad{cmd_flag}")
        return 1

    raw_images = list(iter_images(cfg.input_root, cfg.subdatasets, cfg.image_extensions))
    prep_images = list(iter_images(target_output, cfg.subdatasets, cfg.image_extensions))

    print(f"Raw files found (subdatasets):          {len(raw_images)}")
    print(f"Preprocessed files found (subdatasets): {len(prep_images)}")

    if len(raw_images) != len(prep_images):
        print("WARNING: File count mismatch between raw and preprocessed datasets!")
    else:
        print("PASSED: File counts match exactly.")

    # Integrity check: every preprocessed image must be mode 'L' and 128x128
    invalid_modes = 0
    invalid_sizes = 0
    invalid_mask_values = 0

    for p in prep_images:
        with Image.open(p) as img:
            if img.mode != "L":
                invalid_modes += 1
            if img.size != (cfg.target_size, cfg.target_size):
                invalid_sizes += 1
            if is_mask(p, target_output, cfg.mask_folder):
                arr = np.asarray(img)
                unique = set(np.unique(arr))
                if not unique.issubset({0, 255}):
                    invalid_mask_values += 1

    print(f"Files with invalid mode (!='L'):        {invalid_modes}")
    print(f"Files with invalid size (!=128x128):     {invalid_sizes}")
    print(f"Masks with unexpected gray levels:      {invalid_mask_values}")

    if invalid_modes == 0 and invalid_sizes == 0 and invalid_mask_values == 0:
        print("PASSED: All preprocessed images meet mode, size, and mask constraints.")
    else:
        print("FAILED: Preprocessed dataset contains invalid images.")

    # Measured size comparison
    raw_size_mb = compute_directory_size_mb(cfg.input_root, cfg.image_extensions)
    prep_size_mb = compute_directory_size_mb(target_output, cfg.image_extensions)
    reduction = ((raw_size_mb - prep_size_mb) / raw_size_mb * 100.0) if raw_size_mb > 0 else 0.0

    print("-" * 60)
    print(f"Raw images size:          {raw_size_mb:.2f} MB")
    print(f"Preprocessed images size: {prep_size_mb:.2f} MB")
    print(f"Disk space reduction:     {reduction:.1f}%")
    print("-" * 60)

    # Visual inspection with Matplotlib
    out_vis_dir = PROJECT_ROOT / "understanding" / "08_verify_mvtec_preprocessing"
    out_vis_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_inverted" if args.inverted else ""
    out_vis_path = out_vis_dir / f"preprocessing_verification{suffix}.png"

    sample_categories = [c for c in ["bottle", "cable", "pill"] if c in cfg.subdatasets][:3]
    samples_to_plot: list[tuple[str, Path, Path | None, Image.Image, Image.Image]] = []

    for cat in sample_categories:
        train_good = cfg.input_root / cat / "train" / "good"
        candidates = sorted(train_good.glob("*.png"))
        if candidates:
            src = candidates[0]
            dst = target_output / src.relative_to(cfg.input_root)
            if dst.exists():
                with Image.open(src) as r_img, Image.open(dst) as p_img:
                    samples_to_plot.append((f"{cat} (train)", src, dst, r_img.copy(), p_img.copy()))

    # Sample defect mask
    for cat in cfg.subdatasets:
        gt_dir = cfg.input_root / cat / "ground_truth"
        candidates = sorted(gt_dir.rglob("*.png"))
        if candidates:
            src = candidates[0]
            dst = target_output / src.relative_to(cfg.input_root)
            if dst.exists():
                with Image.open(src) as r_img, Image.open(dst) as p_img:
                    samples_to_plot.append((f"{cat} (mask)", src, dst, r_img.copy(), p_img.copy()))
            break

    # Synthetic non-square example
    synth_raw = Image.new("RGB", (240, 120), color=(220, 180, 60))
    synth_gray = np.asarray(synth_raw.convert("L"), dtype=np.uint8)
    if args.inverted:
        synth_gray = invert_intensity(synth_gray)
    synth_letterboxed = letterbox_to_square(synth_gray, size=128, pad_value=0)
    synth_prep = Image.fromarray(synth_letterboxed, mode="L")
    label_prefix = "Synthetic (Inverted)" if args.inverted else "Synthetic"
    samples_to_plot.append((f"{label_prefix} [120x240]", Path("synthetic_raw.png"), None, synth_raw, synth_prep))

    num_samples = len(samples_to_plot)
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2.5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (title, _, _, r_img, p_img) in enumerate(samples_to_plot):
        axes[i, 0].imshow(r_img, cmap="gray" if r_img.mode == "L" else None)
        axes[i, 0].set_title(f"Raw: {title} ({r_img.size[0]}x{r_img.size[1]}, mode {r_img.mode})", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(p_img, cmap="gray")
        axes[i, 1].set_title(f"Preprocessed ({p_img.size[0]}x{p_img.size[1]}, mode {p_img.mode})", fontsize=9)
        axes[i, 1].axis("off")

    plt.tight_layout()
    fig.savefig(out_vis_path, dpi=150)
    plt.close(fig)
    print(f"Visual comparison saved to: {out_vis_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
