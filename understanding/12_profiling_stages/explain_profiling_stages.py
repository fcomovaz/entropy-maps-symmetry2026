"""Educational step-by-step profiling laboratory (Understanding-First Rule).

For a single 128x128 grayscale image (synthetic structured pattern by default, or loaded via
--image), inspects and contrasts three profiling methodologies on CPU (threads=1):
1. Manual nanosecond stage timing (`time.perf_counter_ns`) for x94 extraction stages:
   image -> quantize_pixels -> region_histograms -> normalized_entropy -> x85 -> asymmetry -> x94
   plus per-layer leaf hooks for `ds_cnn`.
2. Deterministic Python profiling via `cProfile` + `pstats` (locator of hotspots, with trace overhead).
3. PyTorch native C++/ATen profiling via `torch.profiler` (kernel/operator breakdown).

Prints stage timings, numerical equivalence checks, overhead gaps, explanations of why the
three tools disagree in absolute numbers, and generates a stacked-bar figure via matplotlib.

Manual execution command (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe understanding/12_profiling_stages/explain_profiling_stages.py
    .\\.venv\\Scripts\\python.exe understanding/12_profiling_stages/explain_profiling_stages.py --image data/preprocessed/KolektorSDD2/test/defect/10002.png
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
from pathlib import Path
import pstats
import statistics
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.asymmetry_maps import (
    compute_asymmetry_features,
    compute_asymmetry_for_grid,
    reconstruct_grids,
)
from src.features.entropy_maps import (
    build_entropy_plan,
    build_quantization_lut,
    build_shannon_term_lut,
    compute_entropy_features,
    normalized_entropy,
    quantize_pixels,
    region_histograms,
)
from src.models.cnn.architectures import create_model
from src.models.cnn.architectures.common import seed_everything


def create_structured_image(size: int = 128) -> np.ndarray:
    """Create a deterministic 128x128 synthetic uint8 grayscale test image."""
    img = np.zeros((size, size), dtype=np.uint8)
    half = size // 2
    img[:half, :half] = 128
    img[:half, half:] = np.tile(np.linspace(0, 255, half, dtype=np.uint8), (half, 1))
    block = 8
    checker = (np.indices((half // block, half // block)).sum(axis=0) % 2) * 255
    img[half:, :half] = np.repeat(
        np.repeat(checker.astype(np.uint8), block, axis=0), block, axis=1
    )
    rng = np.random.default_rng(42)
    img[half:, half:] = rng.integers(0, 256, (half, half), dtype=np.uint8)
    return img


def load_or_create_image(image_path: Path | None, size: int = 128) -> np.ndarray:
    """Load grayscale image resized to square or generate deterministic synthetic pattern."""
    if image_path is None:
        return create_structured_image(size=size)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as im:
        gray = im.convert("L").resize((size, size), resample=Image.Resampling.BILINEAR)
        return np.asarray(gray, dtype=np.uint8)


def _scale_term_lut(plan_regions: tuple[object, ...], g: int) -> np.ndarray | None:
    """Return shared Shannon term LUT for scale g when all g*g regions have equal pixel counts."""
    regs_g = [r for r in plan_regions if getattr(r, "g") == g]
    if not regs_g:
        return None
    n_pix0 = getattr(regs_g[0], "n_pixels")
    if all(getattr(r, "n_pixels") == n_pix0 for r in regs_g):
        return getattr(regs_g[0], "term_lut")
    return None


def _run_x94_e2e_single(
    img_batch: np.ndarray,
    grid_levels: tuple[int, ...],
    bins: int,
    asym_scales: tuple[int, ...],
    asym_kinds: tuple[str, ...],
    plan: object,
) -> np.ndarray:
    """Execute end-to-end x94 extraction using formal src.features entry points."""
    x85 = compute_entropy_features(img_batch, grid_levels=grid_levels, bins=bins, plan=plan)  # type: ignore[arg-type]
    e_by_scale: dict[int, np.ndarray] = {}
    offset = 0
    for g in grid_levels:
        cols = g * g
        e_by_scale[g] = x85[:, offset : offset + cols]
        offset += cols
    x9 = compute_asymmetry_features(e_by_scale, scales=asym_scales, kinds=asym_kinds)
    return np.concatenate([x85, x9], axis=1)


def _measure_single_x94_stages(
    img_u8: np.ndarray,
    repeats: int = 150,
    warmup: int = 20,
) -> tuple[dict[str, float], float, float, float]:
    """Measure x94 initialization cost, individual stage medians, end-to-end median, and numeric diff."""
    size = int(img_u8.shape[0])
    grid_levels = (1, 2, 4, 8)
    bins = 16
    asym_scales = (2, 4, 8)
    asym_kinds = ("lr", "tb", "rot180")
    img_batch = img_u8[None, ...]

    # 1. Measure one-time plan/LUT precomputation separately
    t0_init = time.perf_counter_ns()
    _ = build_quantization_lut(bins)
    _ = build_shannon_term_lut((size // 2) * (size // 2))
    plan = build_entropy_plan(size, grid_levels, bins)
    init_ms = (time.perf_counter_ns() - t0_init) / 1e6

    scale_term_luts = {g: _scale_term_lut(plan.regions, g) for g in grid_levels}

    stage_names = [
        "1_quantize_pixels",
        "2_hist_g1",
        "2_hist_g2",
        "2_hist_g4",
        "2_hist_g8",
        "3_entropy_g1",
        "3_entropy_g2",
        "3_entropy_g4",
        "3_entropy_g8",
        "4_concat_x85",
        "5_asymmetry_x9",
        "6_concat_x94",
    ]
    stage_samples_ms: dict[str, list[float]] = {name: [] for name in stage_names}
    e2e_samples_ms: list[float] = []

    def _run_staged_once(record: bool) -> np.ndarray:
        t0 = time.perf_counter_ns()
        q = quantize_pixels(img_batch, bins=bins, lut=plan.quant_lut)
        t1 = time.perf_counter_ns()
        if record:
            stage_samples_ms["1_quantize_pixels"].append((t1 - t0) / 1e6)

        e_parts: list[np.ndarray] = []
        e_by_scale: dict[int, np.ndarray] = {}
        for g in grid_levels:
            th0 = time.perf_counter_ns()
            counts_g = region_histograms(q, g=g, bins=bins)
            th1 = time.perf_counter_ns()
            if record:
                stage_samples_ms[f"2_hist_g{g}"].append((th1 - th0) / 1e6)

            te0 = time.perf_counter_ns()
            ent_g = normalized_entropy(counts_g, bins=bins, term_lut=scale_term_luts[g])
            te1 = time.perf_counter_ns()
            if record:
                stage_samples_ms[f"3_entropy_g{g}"].append((te1 - te0) / 1e6)

            e_parts.append(ent_g)
            e_by_scale[g] = ent_g

        tc0 = time.perf_counter_ns()
        x85 = np.concatenate(e_parts, axis=1)
        tc1 = time.perf_counter_ns()
        if record:
            stage_samples_ms["4_concat_x85"].append((tc1 - tc0) / 1e6)

        ta0 = time.perf_counter_ns()
        asym_cols: list[np.ndarray] = []
        for g in asym_scales:
            grids_g = reconstruct_grids(e_by_scale[g], g)
            asym_dict = compute_asymmetry_for_grid(grids_g, kinds=asym_kinds)
            for k in asym_kinds:
                asym_cols.append(asym_dict[k][:, None])
        x9 = np.concatenate(asym_cols, axis=1)
        ta1 = time.perf_counter_ns()
        if record:
            stage_samples_ms["5_asymmetry_x9"].append((ta1 - ta0) / 1e6)

        tf0 = time.perf_counter_ns()
        x94 = np.concatenate([x85, x9], axis=1)
        tf1 = time.perf_counter_ns()
        if record:
            stage_samples_ms["6_concat_x94"].append((tf1 - tf0) / 1e6)

        return x94

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(warmup):
            _ = _run_staged_once(record=False)
            _ = _run_x94_e2e_single(img_batch, grid_levels, bins, asym_scales, asym_kinds, plan)

        x94_staged = np.empty((1, 94), dtype=np.float64)
        x94_e2e = np.empty((1, 94), dtype=np.float64)
        for _ in range(repeats):
            x94_staged = _run_staged_once(record=True)

            t_e0 = time.perf_counter_ns()
            x94_e2e = _run_x94_e2e_single(
                img_batch, grid_levels, bins, asym_scales, asym_kinds, plan
            )
            t_e1 = time.perf_counter_ns()
            e2e_samples_ms.append((t_e1 - t_e0) / 1e6)
    finally:
        if gc_was_enabled:
            gc.enable()

    max_abs_diff = float(np.max(np.abs(x94_staged - x94_e2e)))
    stage_medians = {k: float(statistics.median(v)) for k, v in stage_samples_ms.items()}
    e2e_median_ms = float(statistics.median(e2e_samples_ms))
    return stage_medians, e2e_median_ms, init_ms, max_abs_diff


def _measure_cnn_layers(
    img_u8: np.ndarray,
    repeats: int = 100,
    warmup: int = 20,
) -> tuple[float, list[dict[str, object]], float, str]:
    """Compare hookless forward, leaf-module hooks, and torch.profiler on ds_cnn (CPU, 1 thread)."""
    torch.set_num_threads(1)
    seed_everything(42)
    model = create_model("ds_cnn", num_classes=2).cpu().eval()

    x = torch.from_numpy((img_u8.astype(np.float32) / 255.0 - 0.5) / 0.5).unsqueeze(0).unsqueeze(0)

    # 1. Hookless baseline
    hookless_ms_list: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                _ = model(x)
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                _ = model(x)
                hookless_ms_list.append((time.perf_counter_ns() - t0) / 1e6)
    finally:
        if gc_was_enabled:
            gc.enable()
    hookless_median_ms = float(statistics.median(hookless_ms_list))

    # 2. Leaf-module forward hooks (accumulates shared modules like DepthwiseSeparableConv.relu)
    leaf_names: dict[nn.Module, str] = {}
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            leaf_names[mod] = name or mod.__class__.__name__

    active_starts: dict[str, int] = {}
    per_iter_layer_ns: dict[str, list[int]] = {n: [] for n in leaf_names.values()}
    iter_accum_ns: dict[str, int] = {n: 0 for n in leaf_names.values()}
    call_counts: dict[str, int] = {n: 0 for n in leaf_names.values()}
    out_shapes: dict[str, tuple[int, ...]] = {}

    def _pre_hook(mod: nn.Module, inp: tuple[object, ...]) -> None:
        m_name = leaf_names[mod]
        active_starts[m_name] = time.perf_counter_ns()

    def _post_hook(mod: nn.Module, inp: tuple[object, ...], out: object) -> None:
        end_ns = time.perf_counter_ns()
        m_name = leaf_names[mod]
        start_ns = active_starts.pop(m_name, end_ns)
        iter_accum_ns[m_name] += end_ns - start_ns
        call_counts[m_name] += 1
        if isinstance(out, torch.Tensor):
            out_shapes[m_name] = tuple(int(d) for d in out.shape)

    handles = []
    for mod in leaf_names:
        handles.append(mod.register_forward_pre_hook(_pre_hook))
        handles.append(mod.register_forward_hook(_post_hook))

    hooked_total_ms_list: list[float] = []
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                for k in iter_accum_ns:
                    iter_accum_ns[k] = 0
                    call_counts[k] = 0
                _ = model(x)
            for _ in range(repeats):
                for k in iter_accum_ns:
                    iter_accum_ns[k] = 0
                    call_counts[k] = 0
                t0 = time.perf_counter_ns()
                _ = model(x)
                hooked_total_ms_list.append((time.perf_counter_ns() - t0) / 1e6)
                for k, ns_val in iter_accum_ns.items():
                    per_iter_layer_ns[k].append(ns_val)
    finally:
        for h in handles:
            h.remove()

    hooked_median_ms = float(statistics.median(hooked_total_ms_list))
    layer_rows: list[dict[str, object]] = []
    sum_layers_ms = 0.0
    for mod, m_name in leaf_names.items():
        med_ms = float(statistics.median(per_iter_layer_ns[m_name])) / 1e6
        sum_layers_ms += med_ms
        layer_rows.append(
            {
                "layer_name": m_name,
                "layer_type": mod.__class__.__name__,
                "calls": call_counts[m_name],
                "output_shape": out_shapes.get(m_name, ()),
                "median_ms": med_ms,
            }
        )

    for r in layer_rows:
        r["share_pct"] = (float(r["median_ms"]) / sum_layers_ms * 100.0) if sum_layers_ms > 0 else 0.0

    # 3. torch.profiler ATen operator view
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        with torch.inference_mode():
            for _ in range(10):
                _ = model(x)
                prof.step()

    aten_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=12)
    return hookless_median_ms, layer_rows, hooked_median_ms, aten_table


def main(argv: list[str] | None = None) -> int:
    """Execute educational profiling lab and save breakdown plot."""
    parser = argparse.ArgumentParser(
        prog="explain_profiling_stages.py",
        description="Understanding-first single-image profiling of x94 descriptor stages and ds_cnn layers.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional path to a real grayscale/RGB image (resized to 128x128). Defaults to deterministic synthetic image.",
    )
    args = parser.parse_args(argv)

    img_path = (PROJECT_ROOT / args.image).resolve() if args.image is not None and not args.image.is_absolute() else args.image
    img_u8 = load_or_create_image(img_path, size=128)

    print("=========================================================================")
    print("LAB 12: PROFILING BY STAGES — X94 DESCRIPTOR & DS_CNN (CPU, THREADS=1)")
    print("=========================================================================")
    print(f"Input image shape={img_u8.shape}, dtype={img_u8.dtype}, source={'synthetic' if img_path is None else img_path}")

    # Part 1: x94 Manual stage timings
    stage_medians, e2e_median_ms, init_ms, max_diff = _measure_single_x94_stages(img_u8)
    sum_stages_ms = sum(stage_medians.values())
    gap_ms = sum_stages_ms - e2e_median_ms

    print(f"\n[x94] Precomputation (EntropyPlan + LUTs, amortized): {init_ms:.4f} ms")
    print(f"[x94] Numerical equivalence check |staged - e2e|_max = {max_diff:.3e}")
    assert max_diff < 1e-12, f"Stage equivalence failed: {max_diff}"

    print("\n--- x94 Stage Breakdown (batch=1, median over 150 runs) ---")
    for name, ms in stage_medians.items():
        pct = (ms / sum_stages_ms * 100.0) if sum_stages_ms > 0 else 0.0
        print(f"  {name:<18s}: {ms:8.4f} ms ({pct:5.1f}%)")
    print(f"  {'SUM OF STAGES':<18s}: {sum_stages_ms:8.4f} ms")
    print(f"  {'E2E FORMAL CALL':<18s}: {e2e_median_ms:8.4f} ms")
    print(f"  {'PYTHON CALL GAP':<18s}: {gap_ms:8.4f} ms (intermediate alloc & dispatch overhead)")

    # Part 2: cProfile comparison on x94
    plan = build_entropy_plan(128, (1, 2, 4, 8), 16)
    img_batch = img_u8[None, ...]
    n_cprof = 150
    profiler = cProfile.Profile()
    t0_cp = time.perf_counter_ns()
    profiler.enable()
    for _ in range(n_cprof):
        _ = _run_x94_e2e_single(img_batch, (1, 2, 4, 8), 16, (2, 4, 8), ("lr", "tb", "rot180"), plan)
    profiler.disable()
    cp_per_call_ms = ((time.perf_counter_ns() - t0_cp) / 1e6) / n_cprof

    s_buf = io.StringIO()
    ps = pstats.Stats(profiler, stream=s_buf).sort_stats("cumtime")
    ps.print_stats(12)
    print(f"\n[x94] cProfile instrumented avg/call = {cp_per_call_ms:.4f} ms vs manual E2E median = {e2e_median_ms:.4f} ms")
    print("Top cumulative calls (cProfile):")
    print(s_buf.getvalue())

    # Part 3: CNN Layer Hooks + torch.profiler
    hookless_ms, layer_rows, hooked_ms, aten_table = _measure_cnn_layers(img_u8)
    hook_overhead_pct = ((hooked_ms - hookless_ms) / hookless_ms * 100.0) if hookless_ms > 0 else 0.0
    print("\n--- ds_cnn Leaf Layer Breakdown (CPU, threads=1, batch=1) ---")
    for r in layer_rows:
        print(
            f"  {str(r['layer_name']):<26s} | {str(r['layer_type']):<16s} | "
            f"calls={int(r['calls'])} | shape={str(r['output_shape']):<18s} | "
            f"{float(r['median_ms']):7.4f} ms ({float(r['share_pct']):5.1f}%)"
        )
    print(
        f"\n[ds_cnn] Hookless median={hookless_ms:.4f} ms | Hooked forward={hooked_ms:.4f} ms "
        f"(hook overhead={hook_overhead_pct:+.1f}%)"
    )
    print("\n[ds_cnn] torch.profiler CPU ATen operators (top 12):")
    print(aten_table)

    print("Why perf_counter, cProfile, hooks, and torch.profiler disagree:")
    print("  1. Manual perf_counter_ns around the un-instrumented function gives true latency.")
    print("  2. cProfile triggers a C trace callback on every Python function entry/exit, inflating micro-calls.")
    print("  3. PyTorch forward hooks add Python per-module timestamps + dict lookups around small Conv/BN/ReLU ops.")
    print("  4. torch.profiler traces C++/ATen dispatcher kernels & allocations; ideal for finding ATen bottlenecks.")

    # Part 4: Save matplotlib visualization
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "profiling_stages_breakdown.png"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=160)
    fig.suptitle(
        "Single-Image Stage & Layer Profiling (CPU, threads=1, batch=1)",
        fontsize=12,
        fontweight="bold",
    )

    # Left: x94 aggregated macro-stages stacked & itemized bar
    hist_total = sum(stage_medians[f"2_hist_g{g}"] for g in (1, 2, 4, 8))
    ent_total = sum(stage_medians[f"3_entropy_g{g}"] for g in (1, 2, 4, 8))
    macro_labels = [
        "quantize_pixels",
        "region_histograms (1,2,4,8)",
        "normalized_entropy (1,2,4,8)",
        "concat x85",
        "asymmetry x9",
        "concat x94",
    ]
    macro_vals = [
        stage_medians["1_quantize_pixels"],
        hist_total,
        ent_total,
        stage_medians["4_concat_x85"],
        stage_medians["5_asymmetry_x9"],
        stage_medians["6_concat_x94"],
    ]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
    left_cum = 0.0
    for lbl, val, col in zip(macro_labels, macro_vals, colors):
        axes[0].barh([0], [val], left=left_cum, color=col, edgecolor="white", label=f"{lbl}: {val:.3f} ms")
        left_cum += val
    axes[0].barh([1], [e2e_median_ms], color="#2b2b2b", label=f"E2E fused call: {e2e_median_ms:.3f} ms")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Staged Sum", "Formal E2E"])
    axes[0].set_xlabel("Median CPU Time (ms)")
    axes[0].set_title(f"x94 Descriptor Stage Breakdown (Gap = {gap_ms:+.3f} ms)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(True, axis="x", linestyle=":", alpha=0.6)

    # Right: ds_cnn grouped by module type
    group_totals: dict[str, float] = {}
    for r in layer_rows:
        ltype = str(r["layer_type"])
        group_totals[ltype] = group_totals.get(ltype, 0.0) + float(r["median_ms"])
    sorted_groups = sorted(group_totals.items(), key=lambda kv: kv[1], reverse=True)
    g_names = [kv[0] for kv in sorted_groups]
    g_vals = [kv[1] for kv in sorted_groups]
    axes[1].barh(g_names[::-1], g_vals[::-1], color="#1f77b4", edgecolor="white")
    for idx, val in enumerate(g_vals[::-1]):
        axes[1].text(val, idx, f" {val:.3f} ms", va="center", fontsize=8)
    axes[1].set_xlabel("Summed Leaf Median Time (ms)")
    axes[1].set_title(f"ds_cnn Leaf Layer Families (Hookless={hookless_ms:.3f} ms)")
    axes[1].grid(True, axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
