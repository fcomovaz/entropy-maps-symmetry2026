"""Benchmark exact 256-bin uint8 histograms without array-computing libraries.

The timed histogram kernels receive prepared pixels. Image loading, grayscale
conversion, patch extraction, and synthetic-data generation are outside timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import sys
import time
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

try:
    from PIL import Image
except ImportError as exc:  # Pillow is the only optional dependency.
    raise SystemExit("Pillow is required to load the experiment-01 image baseline") from exc


LEVELS = 256
PATCH_SIZES = (16, 32, 64, 128, 256)
INPUT_TYPES = (
    "uniform",
    "two-level",
    "gradient",
    "checkerboard",
    "low-entropy random",
    "full-range random",
    "real image patch",
)
SEED = 20260928
INDEPENDENT_RUNS = 5
WARMUP_CALLS = 3
TARGET_PIXELS_PER_RUN = 1_000_000
MIN_REPETITIONS = 7
MAX_REPETITIONS = 2_000


HistogramFunction = Callable[[bytes, Image.Image | None], list[int]]


@dataclass(frozen=True)
class Method:
    name: str
    family: str
    function: HistogramFunction
    conceptual_memory: str
    hashing: bool
    dynamic_allocation: bool
    c_ease: str
    embedded_candidate: bool


@dataclass(frozen=True)
class PreparedInput:
    input_type: str
    size: int
    pixels: bytes
    image: Image.Image


def histogram_direct(pixels: bytes, _image: Image.Image | None = None) -> list[int]:
    """H1: direct-address table, the fundamental scalar baseline."""
    hist = [0] * LEVELS
    for pixel in pixels:
        hist[pixel] += 1
    return hist


def histogram_direct_local(pixels: bytes, _image: Image.Image | None = None) -> list[int]:
    """H1-local: same algorithm with explicit local aliases (Python-only check)."""
    hist = [0] * LEVELS
    local_hist = hist
    for pixel in pixels:
        local_hist[pixel] += 1
    return hist


def histogram_array(pixels: bytes, _image: Image.Image | None = None) -> list[int]:
    """H2: fixed-width array.array counters, conceptually close to a C array."""
    hist = array("I", [0]) * LEVELS
    for pixel in pixels:
        hist[pixel] += 1
    return list(hist)


def histogram_counter(pixels: bytes, _image: Image.Image | None = None) -> list[int]:
    """H3: hash/mapping-based Counter comparison."""
    counts = Counter(pixels)
    return [counts[level] for level in range(LEVELS)]


def histogram_defaultdict(pixels: bytes, _image: Image.Image | None = None) -> list[int]:
    """H4: mapping/hash negative control."""
    counts: defaultdict[int, int] = defaultdict(int)
    for pixel in pixels:
        counts[pixel] += 1
    return [counts[level] for level in range(LEVELS)]


def histogram_unroll4(pixels: bytes, _image: Image.Image | None = None) -> list[int]:
    """H5: scalar loop unrolled four times; tail remains exact."""
    hist = [0] * LEVELS
    stop = len(pixels) - len(pixels) % 4
    for index in range(0, stop, 4):
        hist[pixels[index]] += 1
        hist[pixels[index + 1]] += 1
        hist[pixels[index + 2]] += 1
        hist[pixels[index + 3]] += 1
    for index in range(stop, len(pixels)):
        hist[pixels[index]] += 1
    return hist


def histogram_unroll8(pixels: bytes, _image: Image.Image | None = None) -> list[int]:
    """H5: scalar loop unrolled eight times; tail remains exact."""
    hist = [0] * LEVELS
    stop = len(pixels) - len(pixels) % 8
    for index in range(0, stop, 8):
        hist[pixels[index]] += 1
        hist[pixels[index + 1]] += 1
        hist[pixels[index + 2]] += 1
        hist[pixels[index + 3]] += 1
        hist[pixels[index + 4]] += 1
        hist[pixels[index + 5]] += 1
        hist[pixels[index + 6]] += 1
        hist[pixels[index + 7]] += 1
    for index in range(stop, len(pixels)):
        hist[pixels[index]] += 1
    return hist


def histogram_pillow(_pixels: bytes, image: Image.Image | None = None) -> list[int]:
    """H6: native library/C baseline; not an embedded implementation candidate."""
    if image is None:
        raise ValueError("Pillow baseline requires a prepared L-mode image")
    return image.histogram()


METHODS = (
    Method("H1 direct-address list", "scalar", histogram_direct, "256 counters", False, False, "very easy", True),
    Method("H1-local direct-address list", "scalar/Python alias", histogram_direct_local, "256 counters", False, False, "very easy; alias has no C significance", True),
    Method("H2 array.array", "fixed-width array", histogram_array, "256 uint32 counters = 1024 B", False, False, "easy", True),
    Method("H3 collections.Counter", "mapping/hash", histogram_counter, "up to 256 mapping entries + overhead", True, True, "poor", False),
    Method("H4 defaultdict", "mapping/hash", histogram_defaultdict, "up to 256 mapping entries + overhead", True, True, "poor", False),
    Method("H5 unroll x4", "scalar unrolled", histogram_unroll4, "256 counters", False, False, "easy", True),
    Method("H5 unroll x8", "scalar unrolled", histogram_unroll8, "256 counters", False, False, "moderate", True),
    Method("H6 Pillow histogram [library/C implementation baseline]", "native library", histogram_pillow, "library-managed", False, False, "not directly transferable", False),
)


def percentile(sorted_values: Sequence[int], fraction: float) -> float:
    """Return a linearly interpolated percentile without third-party packages."""
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def synthetic_pixels(input_type: str, size: int, seed: int) -> bytes:
    count = size * size
    rng = random.Random(seed)
    if input_type == "uniform":
        return bytes([127]) * count
    if input_type == "two-level":
        return bytes(32 if index % 2 == 0 else 224 for index in range(count))
    if input_type == "gradient":
        return bytes(round(255 * column / max(1, size - 1)) for _row in range(size) for column in range(size))
    if input_type == "checkerboard":
        return bytes(0 if (row + column) % 2 == 0 else 255 for row in range(size) for column in range(size))
    if input_type == "low-entropy random":
        values = (24, 24, 24, 24, 25, 25, 26, 27)
        return bytes(values[rng.randrange(len(values))] for _ in range(count))
    if input_type == "full-range random":
        return bytes(rng.randrange(LEVELS) for _ in range(count))
    raise ValueError(f"Unknown synthetic input: {input_type}")


def load_real_image(experiment_root: Path) -> Image.Image:
    candidates = (
        experiment_root.parent / "01_patch_partition" / "assets" / "derived_grayscale_500x333.png",
        experiment_root.parent / "01_patch_partition" / "assets" / "picsum_entropy-aware-patches_512x512.jpg",
    )
    for path in candidates:
        if path.exists():
            with Image.open(path) as source:
                return source.convert("L")
    raise FileNotFoundError(
        "Run code/understanding/01_patch_partition first; no real source image was found"
    )


def prepare_inputs(experiment_root: Path) -> list[PreparedInput]:
    real_image = load_real_image(experiment_root)
    prepared: list[PreparedInput] = []
    for size in PATCH_SIZES:
        for input_index, input_type in enumerate(INPUT_TYPES):
            if input_type == "real image patch":
                left = (real_image.width - size) // 2
                top = (real_image.height - size) // 2
                image = real_image.crop((left, top, left + size, top + size))
                pixels = image.tobytes()
            else:
                pixels = synthetic_pixels(input_type, size, SEED + size * 100 + input_index)
                image = Image.frombytes("L", (size, size), pixels)
            assert len(pixels) == size * size
            prepared.append(PreparedInput(input_type, size, pixels, image))
    return prepared


def validate_methods(prepared_inputs: Sequence[PreparedInput]) -> None:
    for item in prepared_inputs:
        reference = histogram_direct(item.pixels)
        assert len(reference) == LEVELS
        assert sum(reference) == len(item.pixels)
        for method in METHODS:
            result = method.function(item.pixels, item.image)
            assert len(result) == LEVELS, (item.input_type, item.size, method.name)
            assert sum(result) == len(item.pixels), (item.input_type, item.size, method.name)
            assert result == reference, (item.input_type, item.size, method.name)


def benchmark_kernel(item: PreparedInput, method: Method, target_pixels: int) -> dict[str, object]:
    repetitions = max(MIN_REPETITIONS, min(MAX_REPETITIONS, math.ceil(target_pixels / len(item.pixels))))
    for _ in range(WARMUP_CALLS):
        method.function(item.pixels, item.image)

    per_call_ns: list[int] = []
    checksum = 0
    for _run in range(INDEPENDENT_RUNS):
        start = time.perf_counter_ns()
        for _repetition in range(repetitions):
            result = method.function(item.pixels, item.image)
            checksum += result[0]
        elapsed = time.perf_counter_ns() - start
        per_call_ns.append(round(elapsed / repetitions))
    per_call_ns.sort()
    median_ns = statistics.median(per_call_ns)
    return {
        "patch_size": f"{item.size}x{item.size}",
        "number_of_pixels": len(item.pixels),
        "input_type": item.input_type,
        "method": method.name,
        "median_ns": round(median_ns),
        "min_ns": min(per_call_ns),
        "p25_ns": round(percentile(per_call_ns, 0.25)),
        "p75_ns": round(percentile(per_call_ns, 0.75)),
        "max_ns": max(per_call_ns),
        "iqr_ns": round(percentile(per_call_ns, 0.75) - percentile(per_call_ns, 0.25)),
        "ns_per_pixel": round(median_ns / len(item.pixels), 4),
        "throughput_pixels_per_s": round(len(item.pixels) * 1_000_000_000 / median_ns),
        "speedup_vs_h1": "",  # Filled once matching H1 timing is known.
        "repetitions_per_run": repetitions,
        "independent_runs": INDEPENDENT_RUNS,
        "conceptual_memory": method.conceptual_memory,
        "requires_hashing": method.hashing,
        "requires_dynamic_allocation": method.dynamic_allocation,
        "c_implementation_ease": method.c_ease,
        "embedded_candidate": method.embedded_candidate,
        "checksum": checksum,
    }


def add_speedups(rows: list[dict[str, object]]) -> None:
    baselines: dict[tuple[str, str], int] = {}
    for row in rows:
        if row["method"] == "H1 direct-address list":
            baselines[(str(row["patch_size"]), str(row["input_type"]))] = int(row["median_ns"])
    for row in rows:
        baseline = baselines[(str(row["patch_size"]), str(row["input_type"]))]
        row["speedup_vs_h1"] = round(baseline / int(row["median_ns"]), 4)


def entropy_from_histogram(histogram: Sequence[int], pixel_count: int) -> float:
    entropy = 0.0
    for count in histogram:
        if count:
            probability = count / pixel_count
            entropy -= probability * math.log2(probability)
    return entropy


def benchmark_end_to_end(real_image: Image.Image) -> list[dict[str, object]]:
    """Time crop -> bytes -> H1 histogram -> entropy, then isolate H1 share."""
    rows: list[dict[str, object]] = []
    size = 64
    boxes = [(x, y, x + size, y + size) for y in range(0, 256, size) for x in range(0, 256, size)]
    repetitions = 50
    for _ in range(WARMUP_CALLS):
        patch = real_image.crop(boxes[0])
        entropy_from_histogram(histogram_direct(patch.tobytes()), size * size)

    pipeline_samples: list[int] = []
    histogram_samples: list[int] = []
    entropy_checksum = 0.0
    prepared_pixels = [real_image.crop(box).tobytes() for box in boxes]
    for _run in range(INDEPENDENT_RUNS):
        start = time.perf_counter_ns()
        for _ in range(repetitions):
            for box in boxes:
                patch = real_image.crop(box)
                pixels = patch.tobytes()
                entropy_checksum += entropy_from_histogram(histogram_direct(pixels), len(pixels))
        pipeline_samples.append(round((time.perf_counter_ns() - start) / repetitions))

        start = time.perf_counter_ns()
        for _ in range(repetitions):
            for pixels in prepared_pixels:
                histogram_direct(pixels)
        histogram_samples.append(round((time.perf_counter_ns() - start) / repetitions))

    pipeline_median = statistics.median(pipeline_samples)
    histogram_median = statistics.median(histogram_samples)
    rows.append({
        "patch_size": "64x64",
        "patches": len(boxes),
        "pixels": len(boxes) * size * size,
        "pipeline_scope": "Pillow crop + tobytes + H1 histogram + entropy",
        "pipeline_median_ns": round(pipeline_median),
        "histogram_only_median_ns": round(histogram_median),
        "histogram_share_percent": round(100 * histogram_median / pipeline_median, 3),
        "independent_runs": INDEPENDENT_RUNS,
        "repetitions_per_run": repetitions,
        "entropy_checksum": round(entropy_checksum, 6),
    })
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_table(rows: Sequence[dict[str, object]]) -> str:
    lines = [
        "| Method | Median ns/pixel across all cases | Median speedup vs H1 | Role |",
        "|---|---:|---:|---|",
    ]
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method.name]
        ns_per_pixel = statistics.median(float(row["ns_per_pixel"]) for row in selected)
        speedup = statistics.median(float(row["speedup_vs_h1"]) for row in selected)
        role = "embedded candidate" if method.embedded_candidate else method.family
        lines.append(f"| {method.name} | {ns_per_pixel:.3f} | {speedup:.3f}x | {role} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-pixels-per-run",
        type=int,
        default=TARGET_PIXELS_PER_RUN,
        help="Approximate pixels processed per independent run/configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_pixels_per_run <= 0:
        raise SystemExit("--target-pixels-per-run must be positive")
    experiment_root = Path(__file__).resolve().parent
    prepared = prepare_inputs(experiment_root)
    validate_methods(prepared)

    rows: list[dict[str, object]] = []
    for item in prepared:
        for method in METHODS:
            rows.append(benchmark_kernel(item, method, args.target_pixels_per_run))
        print(f"benchmarked {item.input_type:20s} {item.size}x{item.size}")
    add_speedups(rows)

    results_dir = experiment_root / "results"
    write_csv(results_dir / "benchmark.csv", rows)
    end_to_end = benchmark_end_to_end(load_real_image(experiment_root))
    write_csv(results_dir / "end_to_end.csv", end_to_end)
    (results_dir / "summary_table.md").write_text(aggregate_table(rows), encoding="utf-8")
    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "pillow": Image.__version__,
        "seed": SEED,
        "patch_sizes": PATCH_SIZES,
        "input_types": INPUT_TYPES,
        "methods": [method.name for method in METHODS],
        "warmup_calls": WARMUP_CALLS,
        "independent_runs": INDEPENDENT_RUNS,
        "target_pixels_per_run": args.target_pixels_per_run,
        "timer": "time.perf_counter_ns",
        "correctness": "all methods exactly matched H1; every histogram summed to N",
        "array_I_itemsize_bytes": array("I").itemsize,
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(prepared)} inputs and wrote {len(rows)} kernel rows to {results_dir}")


if __name__ == "__main__":
    main()
