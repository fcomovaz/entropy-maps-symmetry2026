"""Run scientific unit tests of x94; all artifacts stay within laboratory 07."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys

sys.dont_write_bytecode = True
from PIL import Image, ImageDraw, __version__ as pillow_version
from compare_descriptors import (extract, compare, top_features, locality, curve_shape,
                                 NAMES, BLOCKS, GROUPS, partition_boxes, entropy_details, CODE)
from experiments import EXPERIMENTS, FILENAMES, cases

ROOT = Path(__file__).resolve().parent


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write full precision numeric values; structured fields are JSON strings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, allow_nan=False) if isinstance(v, (dict, list, tuple)) else v
                         for k, v in row.items()} for row in rows)


def digest(image: Image.Image) -> str:
    """SHA256 of L pixels; dimensions are stored separately."""
    return hashlib.sha256(image.tobytes()).hexdigest()


def save_overlay(image: Image.Image, path: Path, boxes: list[tuple], grid: int = 8, boundary: int | None = None) -> None:
    """Actual canonical cell borders in red, defect in green, boundary in cyan."""
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for i, (x0, y0, x1, y1) in enumerate(partition_boxes(*image.size, grid)):
        draw.rectangle((x0, y0, x1-1, y1-1), outline="#ed5252")
        draw.text((x0+2, y0+2), f"{i//grid},{i%grid}", fill="yellow", stroke_width=1, stroke_fill="black")
    for x0, y0, x1, y1 in boxes:
        draw.rectangle((x0, y0, x1-1, y1-1), outline="#35ff69", width=2)
    if boundary is not None:
        draw.line((boundary, 0, boundary, image.height-1), fill="cyan", width=1)
    canvas.save(path)


def plot_rows(rows: list[dict], directory: Path, experiment: str) -> None:
    """Optional matplotlib curves: distances, signed entropy means, feature deltas."""
    if not importlib.util.find_spec("matplotlib"):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    for series in dict.fromkeys(row["series"] for row in rows):
        samples = [r for r in rows if r["series"] == series and r["relation"] == "base"]
        if len(samples) < 2:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        xs = [r["parameter"] for r in samples]
        for group in ("all", "entropy", "asymmetry"):
            axes[0, 0].plot(xs, [r[f"{group}_l2"] for r in samples], "o-", label=group)
        for block in BLOCKS:
            axes[0, 1].plot(xs, [r[f"{block}_mad"] for r in samples], ".-", label=block)
        for g in (1, 2, 4, 8):
            axes[1, 0].plot(xs, [r[f"H_{g}x{g}_signed_mean"] for r in samples], "o-", label=f"H_{g}x{g}")
        steps = [r for r in rows if r["series"] == series and r["relation"] == "consecutive"]
        if steps:
            for block in ("all", "H_2x2", "H_4x4", "H_8x8"):
                axes[1, 1].plot([r["parameter"] for r in steps], [r[f"{block}_l2"] for r in steps], ".-", label=block)
            axes[1, 1].set_ylabel("Consecutive L2 (endpoint x)")
        else:
            worst = max(samples, key=lambda r: r["all_l2"])
            axes[1, 1].plot(range(94), worst["delta"], ".", label="delta, largest L2 case")
            axes[1, 1].set_xlabel("Canonical feature index")
            axes[1, 1].set_ylabel("Signed feature difference")
        for ax in axes.flat:
            ax.legend(fontsize=7)
            ax.grid(alpha=.25)
        axes[0, 0].set_ylabel("Distance from base (L2)")
        axes[0, 1].set_ylabel("Block MAD")
        axes[1, 0].set_ylabel("Signed mean entropy change")
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set_xlabel(series + " parameter")
        fig.suptitle(f"{experiment}: {series} (sampled response; lines are guides)")
        fig.tight_layout()
        fig.savefig(directory / f"{experiment}_{series}.png", dpi=130)
        plt.close(fig)


def comparison_row(case, a, b, metadata, tolerance, relation="base") -> dict:
    """Keep original vectors, signed deltas, top dimensions and block statistics."""
    row = {**metadata, "series": case.series, "parameter": case.parameter, "relation": relation,
           "parameters": case.parameters, "base_bins": case.base_bins, "bins": case.bins,
           "base_width": case.base.width, "base_height": case.base.height,
           "width": case.image.width, "height": case.image.height, "tolerance": tolerance,
           "base_vector": a, "modified_vector": b, "delta": [v-u for u, v in zip(a, b)],
           "top10": top_features(a, b, tolerance), **compare(a, b, tolerance)}
    for group, indices in GROUPS.items():
        row[f"{group}_signed_mean"] = math.fsum(b[i]-a[i] for i in indices)/len(indices)
        row[f"{group}_modified_mean"] = math.fsum(b[i] for i in indices)/len(indices)
    return row


def source_hashes() -> dict:
    """Fingerprint the experiment and actual descriptor dependencies."""
    paths = list(ROOT.glob("*.py")) + list((ROOT / "tests").glob("*.py"))
    for lab in ("03_entropy_benchmark", "05_symmetry_features", "06_entropy_pyramid_descriptor"):
        paths.extend((ROOT.parent / lab).glob("*.py"))
    return {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run_experiment(name: str, args, destination: Path) -> list[dict]:
    """Execute one sweep with optional consecutive comparisons and exact artifacts."""
    image_dir = destination / "outputs" / "images" / name
    image_dir.mkdir(parents=True, exist_ok=True)
    rows, previous = [], {}
    cache = {}
    def vector(image, bins):
        key = (image.size, digest(image), bins)
        if key not in cache:
            cache[key] = extract(image, bins)
        return cache[key]
    for index, case in enumerate(cases(name, args.size, args.seed, args.bins)):
        tag = f"{index:03d}_{case.series}"
        base_path, modified_path = image_dir / f"{tag}_base.png", image_dir / f"{tag}_modified.png"
        case.base.save(base_path)
        case.image.save(modified_path)
        save_overlay(case.image, image_dir / f"{tag}_grid.png", case.boxes,
                     case.parameters.get("grid", 8), case.parameters.get("boundary"))
        a, b = vector(case.base, case.base_bins), vector(case.image, case.bins)
        metadata = {"experiment": name, "case_id": tag, "seed": args.seed, "schema": "x94-v1",
                    "base_image": base_path.relative_to(destination).as_posix(),
                    "modified_image": modified_path.relative_to(destination).as_posix(),
                    "base_sha256": digest(case.base), "modified_sha256": digest(case.image),
                    "python": platform.python_version(), "pillow": pillow_version}
        row = comparison_row(case, a, b, metadata, args.tolerance)
        row["extras"] = case.extras
        row["defect_boxes"] = case.boxes
        if case.boxes:
            row.update(locality(case.image.size, case.boxes, a, b, args.tolerance))
        rows.append(row)
        if case.consecutive and case.series in previous:
            previous_case, previous_vector, previous_path = previous[case.series]
            step_metadata = {**metadata, "base_image": previous_path.relative_to(destination).as_posix(),
                             "base_sha256": digest(previous_case.image)}
            step = comparison_row(case, previous_vector, b, step_metadata, args.tolerance, "consecutive")
            step["previous_parameter"] = previous_case.parameter
            step["parameter_step"] = case.parameter-previous_case.parameter
            step["l2_per_parameter_unit"] = step["all_l2"]/abs(step["parameter_step"])
            step["defect_boxes"] = previous_case.boxes+case.boxes
            step.update(locality(case.image.size, step["defect_boxes"], previous_vector, b, args.tolerance))
            rows.append(step)
        previous[case.series] = (case, b, modified_path)
    for series in previous:
        steps = [r for r in rows if r["series"] == series and r["relation"] == "consecutive"]
        if steps:
            med = statistics.median(r["all_l2"] for r in steps)
            maximum = max(r["all_l2"] for r in steps)
            for row in steps:
                row["step_median_l2"] = med
                row["step_to_median_ratio"] = row["all_l2"]/med if med > args.tolerance else None
                row["largest_step"] = abs(row["all_l2"]-maximum) <= args.tolerance
                row["peak_over_3x_median"] = row["all_l2"] > max(3*med, args.tolerance)
    filename = FILENAMES.get(name, name)
    write_csv(destination / "results" / f"{filename}.csv", rows)
    (destination / "results" / f"{filename}_manifest.json").write_text(json.dumps({
        "experiment": name, "size": args.size, "seed": args.seed, "bins": args.bins,
        "tolerance": args.tolerance, "feature_names": NAMES, "source_sha256": source_hashes(),
        "python": platform.python_version(), "pillow": pillow_version, "rows": len(rows),
        "matplotlib_available": bool(importlib.util.find_spec("matplotlib"))}, indent=2), encoding="utf-8")
    plot_rows(rows, destination / "outputs" / "plots", name)
    return rows


def save_summaries(all_rows: list[dict], destination: Path) -> None:
    """Measured response trends, extrema, named changes and multiscale table."""
    summaries, top = [], []
    keys = dict.fromkeys((r["experiment"], r["series"], r["relation"]) for r in all_rows)
    lines = ["# Measured response curves", "", "Generated by run_all.py; descriptive sampled trends only.", ""]
    for experiment, series, relation in keys:
        group = [r for r in all_rows if (r["experiment"], r["series"], r["relation"]) == (experiment, series, relation)]
        worst = max(group, key=lambda r: r["all_l2"])
        summaries.append({"experiment": experiment, "series": series, "relation": relation, "n": len(group),
                          "l2_shape": curve_shape([r["all_l2"] for r in group]),
                          "entropy_mean_shape": curve_shape([r["entropy_modified_mean"] for r in group]),
                          "min_l2": min(r["all_l2"] for r in group), "max_l2": worst["all_l2"],
                          "max_case": worst["case_id"], "max_parameter": worst["parameter"],
                          "max_l1": max(r["all_l1"] for r in group),
                          "max_feature_difference": max(r["all_max_abs"] for r in group),
                          "max_changed": max(r["all_changed"] for r in group)})
        lines += [f"## {experiment} / {series} / {relation}", "",
                  f"L2 range {summaries[-1]['min_l2']:.9g} .. {worst['all_l2']:.9g}; "
                  f"{summaries[-1]['l2_shape']}. Largest case: {worst['case_id']}, parameter {worst['parameter']}.", "",
                  "| Feature | Signed delta |", "|---|---:|"]
        for feature in worst["top10"]:
            lines.append(f"| {feature['name']} | {feature['delta']:.9g} |")
            top.append({"experiment": experiment, "series": series, "relation": relation,
                        "case_id": worst["case_id"], **feature})
        if not worst["top10"]:
            lines.append("| No features exceed tolerance | 0 |")
        lines.append("")
    write_csv(destination / "results" / "response_curves.csv", summaries)
    write_csv(destination / "results" / "top_features.csv", top)
    multiscale = [{"defect_size": r["parameter"], **{f"{block}_{metric}": r[f"{block}_{metric}"]
                  for block in BLOCKS for metric in ("l1", "mad", "l2", "max_abs", "changed")}}
                  for r in all_rows if r["experiment"] == "defect_size" and r["relation"] == "base"]
    if multiscale:
        write_csv(destination / "results" / "multiscale_response.csv", multiscale)
    (destination / "results" / "observations.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all",)+EXPERIMENTS, default="all")
    parser.add_argument("--size", type=int, default=256, help="Square resolution >=128; nonmultiples of 8 are permitted")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--bins", type=int, default=16, help="Canonical 06 quantization, integer 2..256")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=ROOT, help="Destination inside lab 07; use a new subdirectory to preserve runs")
    args = parser.parse_args()
    if args.size < 128 or not 2 <= args.bins <= 256 or not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("Require size >=128, bins 2..256, finite tolerance >=0")
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT):
        parser.error("--output must remain within 07_descriptor_behavior")
    selected = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    all_rows = []
    for name in selected:
        rows = run_experiment(name, args, destination)
        all_rows.extend(rows)
        print(f"{name}: {len(rows)} comparisons saved", flush=True)
    # Aggregate names identify partial runs, so they cannot silently replace full summaries.
    if args.experiment == "all":
        save_summaries(all_rows, destination)
    print(f"Saved {len(all_rows)} comparisons to {destination}")


if __name__ == "__main__":
    main()
