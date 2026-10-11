"""Visual, dependency-light study of non-overlapping image patches.

Pillow is used only for image I/O, cropping, drawing, and composition. Patch
coordinates are computed explicitly with Python integer arithmetic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

from PIL import Image, ImageChops, ImageDraw, ImageFont


SOURCE_URL = "https://picsum.photos/seed/entropy-aware-patches/512/512"
SOURCE_NAME = "picsum_entropy-aware-patches_512x512.jpg"
DIVISIBLE_PATCH_SIZES = (32, 64, 128)
AWKWARD_SIZE = (500, 333)
EdgePolicy = Literal["drop", "partial", "pad"]


@dataclass(frozen=True)
class PatchBox:
    patch_id: int
    row: int
    column: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop_box(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1

    def record(self) -> dict[str, int]:
        data = asdict(self)
        data.update(width=self.width, height=self.height)
        return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_source_image(assets_dir: Path) -> tuple[Path, dict[str, object]]:
    """Download the fixed source once, then validate and reuse it offline."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    image_path = assets_dir / SOURCE_NAME
    if not image_path.exists():
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "patch-partition-study/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            image_path.write_bytes(response.read())

    with Image.open(image_path) as image:
        resolution = [image.width, image.height]
    if resolution != [512, 512]:
        raise ValueError(f"Expected a 512x512 source, received {resolution}")

    metadata = {
        "original_url": SOURCE_URL,
        "local_file": image_path.name,
        "resolution": resolution,
        "sha256": sha256(image_path),
    }
    (assets_dir / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return image_path, metadata


def patch_boxes(
    width: int, height: int, patch_width: int, patch_height: int, policy: EdgePolicy
) -> tuple[list[PatchBox], tuple[int, int]]:
    """Compute row-major patch coordinates manually; x1/y1 are exclusive."""
    if min(width, height, patch_width, patch_height) <= 0:
        raise ValueError("Image and patch dimensions must be positive")

    if policy == "drop":
        columns = width // patch_width
        rows = height // patch_height
        canvas_size = (width, height)
    elif policy in ("partial", "pad"):
        columns = math.ceil(width / patch_width)
        rows = math.ceil(height / patch_height)
        canvas_size = (
            columns * patch_width if policy == "pad" else width,
            rows * patch_height if policy == "pad" else height,
        )
    else:
        raise ValueError(f"Unknown edge policy: {policy}")

    boxes: list[PatchBox] = []
    for row in range(rows):
        y0 = row * patch_height
        y1 = y0 + patch_height
        for column in range(columns):
            x0 = column * patch_width
            x1 = x0 + patch_width
            if policy == "partial":
                x1 = min(x1, width)
                y1 = min(y1, height)
            boxes.append(PatchBox(len(boxes), row, column, x0, y0, x1, y1))
    return boxes, canvas_size


def validate_boxes(
    boxes: list[PatchBox], canvas_size: tuple[int, int], patch_size: tuple[int, int], policy: EdgePolicy
) -> None:
    width, height = canvas_size
    patch_width, patch_height = patch_size
    for box in boxes:
        assert 0 <= box.x0 < box.x1 <= width
        assert 0 <= box.y0 < box.y1 <= height
        if policy != "partial":
            assert (box.width, box.height) == patch_size
    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            overlap_width = min(first.x1, second.x1) - max(first.x0, second.x0)
            overlap_height = min(first.y1, second.y1) - max(first.y0, second.y0)
            assert overlap_width <= 0 or overlap_height <= 0, (first, second)
    expected_columns = width // patch_width if policy == "drop" else math.ceil(width / patch_width)
    expected_rows = height // patch_height if policy == "drop" else math.ceil(height / patch_height)
    assert len(boxes) == expected_columns * expected_rows


def prepare_canvas(image: Image.Image, canvas_size: tuple[int, int], fill: int = 0) -> Image.Image:
    if image.size == canvas_size:
        return image.copy()
    canvas = Image.new("L", canvas_size, color=fill)
    canvas.paste(image, (0, 0))
    return canvas


def reconstruct(patches: Iterable[tuple[PatchBox, Image.Image]], canvas_size: tuple[int, int]) -> Image.Image:
    reconstruction = Image.new("L", canvas_size, color=0)
    for box, patch in patches:
        assert patch.size == (box.width, box.height)
        reconstruction.paste(patch, (box.x0, box.y0))
    return reconstruction


def draw_grid(image: Image.Image, boxes: list[PatchBox], original_size: tuple[int, int]) -> Image.Image:
    visual = image.convert("RGB")
    draw = ImageDraw.Draw(visual)
    original_width, original_height = original_size
    if image.size != original_size:
        draw.rectangle((0, 0, original_width - 1, original_height - 1), outline=(0, 180, 255), width=3)
    for box in boxes:
        draw.rectangle((box.x0, box.y0, box.x1 - 1, box.y1 - 1), outline=(255, 55, 40), width=1)
    return visual


def contact_sheet(patches: list[tuple[PatchBox, Image.Image]]) -> Image.Image:
    font = ImageFont.load_default()
    tile = 96
    label_height = 18
    columns = min(16, max(1, math.ceil(math.sqrt(len(patches)))))
    rows = math.ceil(len(patches) / columns)
    sheet = Image.new("RGB", (columns * tile, rows * (tile + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (box, patch) in enumerate(patches):
        left = (index % columns) * tile
        top = (index // columns) * (tile + label_height)
        preview = patch.convert("RGB")
        preview.thumbnail((tile, tile))
        offset_x = left + (tile - preview.width) // 2
        offset_y = top + (tile - preview.height) // 2
        sheet.paste(preview, (offset_x, offset_y))
        draw.rectangle((left, top, left + tile - 1, top + tile - 1), outline="black")
        draw.text((left + 2, top + tile + 2), f"ID {box.patch_id} r{box.row} c{box.column}", fill="black", font=font)
    return sheet


def save_manifest(boxes: list[PatchBox], path: Path) -> None:
    fields = ["patch_id", "row", "column", "x0", "y0", "x1", "y1", "width", "height"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(box.record() for box in boxes)


def selected_sample_ids(count: int) -> set[int]:
    return {0, min(1, count - 1), count // 2, max(0, count - 2), count - 1}


def run_case(
    image: Image.Image, patch_size: int, policy: EdgePolicy, output_dir: Path, case_name: str
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    boxes, canvas_size = patch_boxes(image.width, image.height, patch_size, patch_size, policy)
    validate_boxes(boxes, canvas_size, (patch_size, patch_size), policy)
    canvas = prepare_canvas(image, canvas_size)
    patches = [(box, canvas.crop(box.crop_box())) for box in boxes]
    rebuilt = reconstruct(patches, canvas_size)
    if policy == "drop":
        expected_reconstruction = Image.new("L", canvas_size, color=0)
        covered_width = max((box.x1 for box in boxes), default=0)
        covered_height = max((box.y1 for box in boxes), default=0)
        expected_reconstruction.paste(canvas.crop((0, 0, covered_width, covered_height)), (0, 0))
    else:
        expected_reconstruction = canvas
    assert ImageChops.difference(expected_reconstruction, rebuilt).getbbox() is None

    canvas.save(output_dir / "partition_input.png")
    draw_grid(canvas, boxes, image.size).save(output_dir / "grid.png")
    contact_sheet(patches).save(output_dir / "contact_sheet.png")
    rebuilt.save(output_dir / "reconstruction.png")
    samples_dir = output_dir / "sample_patches"
    samples_dir.mkdir(exist_ok=True)
    sample_ids = selected_sample_ids(len(patches))
    for box, patch in patches:
        if box.patch_id in sample_ids:
            patch.save(samples_dir / f"patch_{box.patch_id:03d}_r{box.row}_c{box.column}.png")
    save_manifest(boxes, output_dir / "patch_coordinates.csv")

    original_pixels = image.width * image.height
    if policy == "drop":
        real_processed = len(boxes) * patch_size * patch_size
        discarded = original_pixels - real_processed
        artificial = 0
    elif policy == "partial":
        real_processed = sum(box.width * box.height for box in boxes)
        discarded = 0
        artificial = 0
    else:
        real_processed = original_pixels
        discarded = 0
        artificial = canvas.width * canvas.height - original_pixels
    embedded_tradeoffs = {
        "drop": {
            "advantage": "Uniform patch buffers without artificial padding.",
            "disadvantage": "Real border pixels are not processed.",
        },
        "partial": {
            "advantage": "All real pixels are covered without artificial padding.",
            "disadvantage": "Border patches require variable-size handling.",
        },
        "pad": {
            "advantage": "Uniform patch buffers while covering all real pixels.",
            "disadvantage": "Artificial pixels must be stored and processed.",
        },
    }
    metrics: dict[str, object] = {
        "case": case_name,
        "edge_policy": policy,
        "original_resolution": list(image.size),
        "partition_canvas_resolution": list(canvas_size),
        "patch_size": [patch_size, patch_size],
        "patches_generated": len(boxes),
        "original_pixels_processed": real_processed,
        "original_pixels_discarded": discarded,
        "artificial_pixels_added": artificial,
        "real_image_coverage_percent": round(100 * real_processed / original_pixels, 6),
        "pixel_perfect_reconstruction_of_generated_patches": True,
        "embedded_system_tradeoffs": embedded_tradeoffs[policy],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def self_test() -> None:
    boxes, canvas = patch_boxes(512, 512, 64, 64, "drop")
    validate_boxes(boxes, canvas, (64, 64), "drop")
    assert len(boxes) == 64 and boxes[-1].crop_box() == (448, 448, 512, 512)

    drop, drop_canvas = patch_boxes(500, 333, 64, 64, "drop")
    assert len(drop) == 35 and drop_canvas == (500, 333)
    assert max(box.x1 for box in drop) == 448 and max(box.y1 for box in drop) == 320
    partial, partial_canvas = patch_boxes(500, 333, 64, 64, "partial")
    assert len(partial) == 48 and partial_canvas == (500, 333)
    assert partial[-1].crop_box() == (448, 320, 500, 333)
    pad, pad_canvas = patch_boxes(500, 333, 64, 64, "pad")
    assert len(pad) == 48 and pad_canvas == (512, 384)
    assert pad[-1].crop_box() == (448, 320, 512, 384)
    for policy, items, size in (
        ("drop", drop, drop_canvas), ("partial", partial, partial_canvas), ("pad", pad, pad_canvas)
    ):
        validate_boxes(items, size, (64, 64), policy)  # type: ignore[arg-type]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize manual grayscale patch partitioning and edge policies."
    )
    parser.add_argument(
        "--patch-size", type=int, default=64, help="Square patch size for the 500x333 edge-policy study (default: 64)."
    )
    parser.add_argument(
        "--edge-policy", choices=("drop", "partial", "pad"), help="Run only this edge policy; omit to run all three."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.patch_size <= 0:
        raise SystemExit("--patch-size must be positive")
    self_test()
    root = Path(__file__).resolve().parent
    source_path, metadata = ensure_source_image(root / "assets")
    with Image.open(source_path) as source:
        grayscale = source.convert("L")
    assert grayscale.size == (512, 512)

    all_metrics: list[dict[str, object]] = []
    for size in DIVISIBLE_PATCH_SIZES:
        all_metrics.append(run_case(grayscale, size, "drop", root / "outputs" / f"A_divisible_patch_{size}", "A_divisible"))

    awkward = grayscale.crop((0, 0, AWKWARD_SIZE[0], AWKWARD_SIZE[1]))
    awkward.save(root / "assets" / "derived_grayscale_500x333.png")
    policies: tuple[EdgePolicy, ...] = (args.edge_policy,) if args.edge_policy else ("drop", "partial", "pad")
    for policy in policies:
        all_metrics.append(
            run_case(awkward, args.patch_size, policy, root / "outputs" / f"B_500x333_patch_{args.patch_size}_{policy}", "B_non_divisible")
        )

    summary = {"source": metadata, "runs": all_metrics}
    (root / "outputs" / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(all_metrics)} cases. Outputs: {root / 'outputs'}")


if __name__ == "__main__":
    main()
