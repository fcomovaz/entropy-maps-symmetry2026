"""Inspect image -> x94, export JSON/CSV and render entropy grids with Pillow."""
import argparse
import csv
import hashlib
from importlib import import_module
import json
from pathlib import Path
import platform
import sys

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

# Numeric experiment folders are importable through importlib. One code/ entry
# supports fully qualified imports without generic module-name collisions.
ROOT = Path(__file__).resolve().parent
if str(ROOT.parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT.parents[1]))
PACKAGE = "understanding.06_entropy_pyramid_descriptor"
api = import_module(PACKAGE)
descriptor_module = import_module(PACKAGE + ".descriptor")
entropy_module = import_module(PACKAGE + ".entropy")
partition_boxes = import_module(PACKAGE + ".partition").partition_boxes
examples_module = import_module(PACKAGE + ".synthetic_examples")


def export_descriptor(result: api.DescriptorResult, path: Path, metadata: dict | None = None) -> None:
    """Write paired .json and .csv files using the explicit feature-name order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {**(metadata or {}), "width": result.image_width,
                     "height": result.image_height, "bins": result.bins,
                     "schema": "x94-v1", "partition": "floor-proportional",
                     "grayscale": "Pillow convert L", "python": platform.python_version(),
                     "pillow": pillow_version},
        "feature_names": result.feature_names, "vector": result.vector,
        "features": dict(zip(result.feature_names, result.vector, strict=True)),
        "entropy_grids": result.entropy_grids,
    }
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("feature_name", "value"))
        writer.writerows(zip(result.feature_names, result.vector, strict=True))


def render_grid(grid: list[list[float]], title: str) -> Image.Image:
    """Draw fixed 0=black, 1=white cells with three-decimal value labels."""
    g, cell, header = len(grid), 72, 40
    side = max(288, g * cell)
    cell = side // g
    canvas = Image.new("RGB", (side, side + header), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    draw.text((8, 10), title, fill="black", font=font)
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            shade = round(255 * value)
            x, y = c * cell, header + r * cell
            draw.rectangle((x, y, x + cell - 1, y + cell - 1),
                           fill=(shade, shade, shade), outline="gray")
            draw.text((x + cell / 2, y + cell / 2), f"{value:.3f}", anchor="mm",
                      fill="white" if value < .5 else "black", font=font)
    return canvas


def image_with_grid(image: Image.Image, g: int) -> Image.Image:
    """Show actual partition boxes with r,c labels; magnify tiny images only for display."""
    factor = max(1, (512 + min(image.size) - 1) // min(image.size))
    canvas = image.convert("RGB").resize((image.width * factor, image.height * factor),
                                         Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    for index, (x0, y0, x1, y1) in enumerate(partition_boxes(*image.size, g)):
        draw.rectangle((x0 * factor, y0 * factor, x1 * factor - 1, y1 * factor - 1),
                       outline="red", width=1)
        draw.text((x0 * factor + 3, y0 * factor + 3), f"{index // g},{index % g}",
                  fill="yellow", stroke_width=1, stroke_fill="black", font=font)
    return canvas


def save_trace(image: Image.Image, result: api.DescriptorResult, path: Path) -> None:
    """Persist the full manual chain including matrix, all boxes and histogram terms."""
    trace = {"matrix": [list(image.crop((0, y, image.width, y + 1)).tobytes())
                        for y in range(image.height)], "levels": {}}
    for g in (1, 2, 4, 8):
        trace["levels"][g] = [
            {"row": i // g, "column": i % g, "box": box,
             **entropy_module.entropy_details(image.crop(box), result.bins)}
            for i, box in enumerate(partition_boxes(*image.size, g))]
    trace.update(entropy_grids=result.entropy_grids, feature_names=result.feature_names,
                 entropy_features=result.entropy_features, asymmetry_features=result.asymmetry_features,
                 vector=result.vector, bins=result.bins)
    path.write_text(json.dumps(trace, indent=2, allow_nan=False), encoding="utf-8")


def print_summary(name: str, result: api.DescriptorResult) -> None:
    """Print dimensions, first five features and all nine asymmetries."""
    print(f"\n{'=' * 50}\nEntropy Pyramid Descriptor: {name}\n{'=' * 50}")
    print(f"Image:\n  size: {result.image_width} x {result.image_height}\n  bins: {result.bins}")
    print("Entropy features:\n  1x1: 1\n  2x2: 4\n  4x4: 16\n  8x8: 64\n  total: 85")
    print("Asymmetry features:\n  2x2: 3\n  4x4: 3\n  8x8: 3\n  total: 9")
    print(f"Descriptor dimensions:\n  {len(result.vector)}\nFirst features:")
    for name, value in zip(result.feature_names[:5], result.vector[:5]):
        print(f"  {name} = {value:.9f}")
    print("Asymmetry:")
    for name, value in zip(api.get_asymmetry_feature_names(), result.asymmetry_features):
        print(f"  {name} = {value:.9f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Input image; omitted: six synthetic cases")
    parser.add_argument("--bins", type=int, default=16, metavar="B", help="Integer 2..256 (default: 16)")
    parser.add_argument("--seed", type=int, default=examples_module.SEED, help="Synthetic noise seed")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs",
                        help="Directory, or .json/.csv filename with --image; always exports both formats")
    args = parser.parse_args()
    try:
        entropy_module.validate_bins(args.bins)
        output = args.output.resolve()
        is_file = output.suffix.lower() in (".json", ".csv")
        if is_file and args.image is None:
            parser.error("A .json/.csv output filename requires --image; use a directory for synthetic cases")
        images = {args.image.stem: descriptor_module.load_grayscale(args.image)} if args.image else examples_module.generate_examples(args.seed)
        out = output.parent if is_file else output
        for folder in ("images", "grids", "descriptors"):
            (out / folder).mkdir(parents=True, exist_ok=True)
        for name, gray in images.items():
            result = api.extract_descriptor(gray, bins=args.bins)
            print_summary(name, result)
            gray.save(out / "images" / f"{name}.png")
            metadata = {"source": str(args.image.resolve()) if args.image else name,
                        "seed": None if args.image else args.seed,
                        "grayscale_sha256": hashlib.sha256(gray.tobytes()).hexdigest()}
            export_descriptor(result, output if is_file else out / "descriptors" / f"{name}.json", metadata)
            grids_out = out / "grids" / name
            grids_out.mkdir(parents=True, exist_ok=True)
            for g in (1, 2, 4, 8):
                render_grid(result.entropy_grids[g], f"Entropy {g}x{g}; B={args.bins}").save(
                    grids_out / f"entropy_grid_{g}x{g}.png")
                image_with_grid(gray, g).save(grids_out / f"image_with_grid_{g}x{g}.png")
            if args.image is None and name == "worked_16x16":
                save_trace(gray, result, out / "descriptors" / "worked_trace.json")
        print(f"\nSaved {len(images)} cases, JSON/CSV and PNGs to {out}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
