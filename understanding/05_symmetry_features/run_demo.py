"""Run matrix examples before images; save visualizations and exact feature vectors."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

from entropy_pyramid import compute_entropy_pyramid, entropy_details, flatten_entropy_features
from symmetry_features import (
    Grid, absolute_difference, compute_nine_asymmetry_features, mirror_left_right,
    mirror_top_bottom, reflection_asymmetry_lr, reflection_asymmetry_tb,
    rot180_asymmetry, rotate_180,
)
from synthetic_images import SEED, generate_images
from synthetic_matrices import controlled_matrices, progressive_defects

ROOT = Path(__file__).resolve().parent
TRANSFORMS = {"lr": mirror_left_right, "tb": mirror_top_bottom, "rot180": rotate_180}


def render_grid(grid: Grid, title: str) -> Image.Image:
    """Render [g,g] values with fixed [0,1] grayscale and numeric cell labels."""
    g, side, header = len(grid), 512, 44
    canvas = Image.new("RGB", (side, side+header), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    draw.text((10, 10), title, fill="black", font=font)
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            shade = round(255*value)
            x, y = j*side//g, header+i*side//g
            draw.rectangle((x, y, (j+1)*side//g-1, header+(i+1)*side//g-1),
                           fill=(shade, shade, shade), outline=(128, 128, 128))
            draw.text((x+side//g/2, y+side//g/2), f"{value:.3f}",
                      anchor="mm", fill="white" if value < .5 else "black", font=font)
    return canvas


def save_comparisons(grid: Grid, directory: Path) -> None:
    """Save original, three transforms and three |E-T(E)| images with fixed scale."""
    directory.mkdir(parents=True, exist_ok=True)
    g = len(grid)
    render_grid(grid, f"Original {g}x{g}").save(directory / f"original_{g}x{g}.png")
    for name, transform in TRANSFORMS.items():
        transformed = transform(grid)
        filename = f"mirror_{name}" if name != "rot180" else name
        render_grid(transformed, f"{filename} {g}x{g}").save(
            directory / f"{filename}_{g}x{g}.png")
        difference = absolute_difference(grid, transformed)
        render_grid(difference, f"abs(E - {name}(E)); scale 0 to 1").save(
            directory / f"difference_{name}_{g}x{g}.png")


def matrix_metrics(grid: Grid) -> dict[str, float]:
    """Return mean absolute entropy-layout residuals in LR,TB,R180 order."""
    return {"lr": reflection_asymmetry_lr(grid), "tb": reflection_asymmetry_tb(grid),
            "rot180": rot180_asymmetry(grid)}


def print_metrics(name: str, grids: dict[int, Grid]) -> None:
    """Print only the actual scales present; do not invent levels for manual matrices."""
    print(f"\n{'='*50}\nExample: {name}\n{'='*50}")
    for g, grid in grids.items():
        print(f"\n{g}x{g}:")
        for key, value in matrix_metrics(grid).items():
            print(f"  {key.upper():7}: {value:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, choices=range(2, 257), default=16,
                        metavar="B", help="Histogram bins, integer 2..256 (default: 16)")
    parser.add_argument("--seed", type=int, default=SEED, help="Balanced-noise shuffle seed")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs",
                        help="Output directory; repeated runs replace matching demo files")
    args = parser.parse_args()
    out = args.output.resolve()
    for folder in ("matrices", "images", "entropy_grids", "differences"):
        (out / folder).mkdir(parents=True, exist_ok=True)

    matrix_results = {}
    matrices = controlled_matrices()
    matrices.update({f"F_defect_{delta:.2f}": grid
                     for delta, grid in progressive_defects().items()})
    for name, grid in matrices.items():
        print_metrics(name, {len(grid): grid})
        matrix_results[name] = {"grid": grid, "asymmetry": matrix_metrics(grid)}
        save_comparisons(grid, out / "matrices" / name)
    (out / "matrices" / "results.json").write_text(
        json.dumps(matrix_results, indent=2), encoding="utf-8")
    with (out / "matrices" / "progressive_defect.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["perturbation", "lr", "tb", "rot180"])
        for delta, grid in progressive_defects().items():
            writer.writerow([delta, *matrix_metrics(grid).values()])

    results = {}
    for name, image in generate_images(args.seed).items():
        image.save(out / "images" / f"{name}.png")
        grids = compute_entropy_pyramid(image, args.bins)
        entropy_features = flatten_entropy_features(grids)
        features = compute_nine_asymmetry_features(grids[2], grids[4], grids[8])
        descriptor = entropy_features + list(features.values())
        assert (len(entropy_features), len(features), len(descriptor)) == (85, 9, 94)
        print_metrics(name, {g: grids[g] for g in (2, 4, 8)})
        print("\n9 asymmetry features:")
        print(list(features))
        print([round(value, 8) for value in features.values()])
        print("Descriptor lengths: 85 entropy + 9 asymmetry = 94")
        directory = out / "entropy_grids" / name
        directory.mkdir(parents=True, exist_ok=True)
        overview = Image.new("RGB", (1024, 1112), "white")
        original = Image.new("RGB", (512, 556), "white")
        ImageDraw.Draw(original).text((10, 10), name, fill="black",
                                     font=ImageFont.load_default(size=16))
        original.paste(image.resize((512, 512), Image.Resampling.NEAREST), (0, 44))
        overview.paste(original, (0, 0))
        for g, position in zip((2, 4, 8), ((512, 0), (0, 556), (512, 556))):
            rendered = render_grid(grids[g], f"Entropy {g}x{g}; H/log2({args.bins})")
            rendered.save(directory / f"entropy_{g}x{g}.png")
            overview.paste(rendered, position)
            save_comparisons(grids[g], out / "differences" / name / f"{g}x{g}")
        overview.save(directory / "overview.png")
        results[name] = {
            "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "entropy_grids": grids, "entropy_features": entropy_features,
            "asymmetry_features": features, "combined_descriptor": descriptor,
            "first_8x8_cell_trace": entropy_details(image.crop((0, 0, 32, 32)), args.bins),
        }
    payload = {"parameters": {"bins": args.bins, "seed": args.seed, "size": [256, 256],
                              "python": platform.python_version(), "pillow": pillow_version,
                              "entropy_order": "levels 1,2,4,8; row-major within level",
                              "asymmetry_order": list(features)}, "images": results}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out / "scale.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["grid", "lr", "tb", "rot180", "max_cell_difference_lr"])
        scale_grids = results["image_08_scale_defect"]["entropy_grids"]
        for g in (2, 4, 8):
            grid = scale_grids[g]
            difference = absolute_difference(grid, mirror_left_right(grid))
            writer.writerow([g, *matrix_metrics(grid).values(), max(map(max, difference))])
    print(f"\nSaved {len(matrices)} matrix cases and {len(results)} images to {out}")


if __name__ == "__main__":
    main()
