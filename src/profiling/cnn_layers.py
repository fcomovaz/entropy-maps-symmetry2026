"""Per-layer leaf hook profiling, torch.profiler ATen breakdown, memory footprint, and thread scaling for CNNs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from src.models.cnn.architectures.common import model_stats
from src.models.cnn.benchmark import estimate_flops
from .timing import (
    LatencyStats,
    disabled_gc,
    measure_indexed_latencies,
    summarize_latencies,
    thread_limits,
)

logger = logging.getLogger("research")


@dataclass(frozen=True, slots=True)
class CnnLayerProfileRecord:
    """Measured latency, parameter/activation memory, FLOPs, and intensity for one leaf layer."""

    model_name: str
    layer_name: str
    layer_type: str
    category: str
    call_count: int
    input_shape: str
    output_shape: str
    params: int
    param_mb_fp32: float
    activation_bytes: int
    activation_kb: float
    flops: int
    median_ms: float
    q25_ms: float
    q75_ms: float
    iqr_ms: float
    p95_ms: float
    share_of_layers_pct: float
    ms_per_mflop: float | None


@dataclass(frozen=True, slots=True)
class AtenOpProfileRecord:
    """Aggregated ATen operator timing, memory, and FLOPs extracted via `torch.profiler`."""

    model_name: str
    op_name: str
    call_count: int
    cpu_time_total_ms: float
    self_cpu_time_ms: float
    share_cpu_time_pct: float
    share_self_cpu_time_pct: float
    cpu_memory_kb: float
    self_cpu_memory_kb: float
    flops: int


@dataclass(frozen=True, slots=True)
class CnnColdStartRecord:
    """Checkpoint deserialization and first-call cold start vs steady-state CPU latencies."""

    model_name: str
    checkpoint_load_ms: float
    first_inference_ms: float
    steady_state_median_ms: float
    cold_to_steady_ratio: float


@dataclass(frozen=True, slots=True)
class CnnLayerProfileReport:
    """Comprehensive CPU profiling report for one CNN architecture."""

    model_name: str
    threads: int
    batch_size: int
    image_size: int
    total_params: int
    trainable_params: int
    model_size_mb: float
    max_activation_kb: float
    total_flops: int
    hookless_stats: LatencyStats
    hooked_total_median_ms: float
    sum_layers_median_ms: float
    hook_overhead_pct: float
    hook_overhead_exceeds_warn: bool
    category_shares_pct: dict[str, float]
    layers: tuple[CnnLayerProfileRecord, ...]
    aten_ops: tuple[AtenOpProfileRecord, ...]
    aten_table_text: str
    chrome_trace_path: Path | None
    cold_start: CnnColdStartRecord | None
    raw_hookless_latencies_ms: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CnnThreadScalingRecord:
    """CPU latency and parallel scaling efficiency across thread counts for one CNN."""

    model_name: str
    threads: int
    batch_size: int
    median_ms: float
    iqr_ms: float
    p95_ms: float
    ci95_low_ms: float
    ci95_high_ms: float
    throughput_samples_s: float
    speedup_vs_1thread: float
    efficiency_pct: float


def _classify_layer(mod: nn.Module) -> str:
    """Classify a PyTorch leaf module into a semantically meaningful operator family."""
    if isinstance(mod, nn.Conv2d):
        if mod.groups == mod.in_channels and mod.in_channels > 1:
            return "dwconv"
        if tuple(mod.kernel_size) == (1, 1) and mod.groups == 1:
            return "pwconv"
        return "conv"
    if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
        return "bn"
    if isinstance(mod, (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.Hardsigmoid, nn.SiLU, nn.GELU)):
        return "activation"
    if isinstance(mod, (nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
        return "pool"
    if isinstance(mod, nn.Linear):
        return "linear"
    return "other"


def _tensor_shape_and_bytes(obj: Any) -> tuple[str, int]:
    """Extract readable shape string and activation byte footprint from tensor or nested output."""
    if isinstance(obj, torch.Tensor):
        shape_str = "x".join(str(int(d)) for d in obj.shape)
        return f"[{shape_str}]", int(obj.numel() * obj.element_size())
    if isinstance(obj, (tuple, list)) and obj:
        shapes = []
        total_b = 0
        for item in obj:
            s_str, b_cnt = _tensor_shape_and_bytes(item)
            shapes.append(s_str)
            total_b += b_cnt
        return "+".join(shapes), total_b
    return str(type(obj).__name__), 0


def _analytical_layer_flops(mod: nn.Module, out: Any) -> int:
    """Estimate Conv2d / Linear forward FLOPs (2 * MACs) matching FlopCounterMode convention."""
    if isinstance(mod, nn.Conv2d) and isinstance(out, torch.Tensor):
        # weight shape: [Cout, Cin // groups, Kh, Kw]; out shape: [B, Cout, Hout, Wout]
        b = int(out.shape[0])
        h_out = int(out.shape[2])
        w_out = int(out.shape[3])
        c_out = int(mod.weight.shape[0])
        kernel_elems = int(mod.weight.shape[1] * mod.weight.shape[2] * mod.weight.shape[3])
        return 2 * b * c_out * h_out * w_out * kernel_elems
    if isinstance(mod, nn.Linear) and isinstance(out, torch.Tensor):
        b = int(out.shape[0]) if out.ndim >= 2 else 1
        return 2 * b * int(mod.in_features) * int(mod.out_features)
    return 0


def measure_cnn_preprocess_latency(
    images_u8: np.ndarray,
    normalize_mean: float,
    normalize_std: float,
    batch_size: int = 1,
    warmup_iters: int = 20,
    timed_iters: int = 200,
    disable_gc: bool = True,
    bootstrap_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> LatencyStats:
    """Measure CPU image-to-normalized-tensor preprocessing latency matching `load_cnn_partition`."""
    n_avail = int(images_u8.shape[0])
    if n_avail == 0:
        raise ValueError("images_u8 must contain at least 1 image")

    batches_u8: list[np.ndarray] = []
    if batch_size == 1:
        for i in range(n_avail):
            batches_u8.append(np.ascontiguousarray(images_u8[i]))
    else:
        n_batches = max(1, (n_avail + batch_size - 1) // batch_size)
        for b in range(n_batches):
            idx = (np.arange(b * batch_size, (b + 1) * batch_size) % n_avail).astype(np.int64)
            batches_u8.append(np.ascontiguousarray(images_u8[idx]))
    n_batches = len(batches_u8)

    def _preprocess_one(idx: int) -> torch.Tensor:
        sample = batches_u8[idx % n_batches]
        arr = sample.astype(np.float32) / 255.0
        arr = (arr - normalize_mean) / normalize_std
        t = torch.from_numpy(arr)
        if sample.ndim == 2:
            return t.unsqueeze(0).unsqueeze(0)
        return t.unsqueeze(1)

    lat_ms = measure_indexed_latencies(
        fn=_preprocess_one,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        disable_gc=disable_gc,
    )
    return summarize_latencies(
        lat_ms,
        per_sample_divisor=batch_size,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )


def measure_checkpoint_cold_start(
    model_factory: Any,
    model_name: str,
    checkpoint_path: Path,
    sample_input: torch.Tensor,
    threads: int = 1,
    steady_warmup: int = 15,
    steady_iters: int = 50,
    disable_gc: bool = True,
) -> CnnColdStartRecord:
    """Measure checkpoint deserialization time, first un-warmed forward pass, and steady state."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    with thread_limits(threads):
        t0_load = time.perf_counter_ns()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = model_factory(model_name, num_classes=2).cpu()
        model.load_state_dict(state)
        model.eval()
        load_ms = (time.perf_counter_ns() - t0_load) / 1e6

        x1 = sample_input[:1].contiguous().cpu()
        with torch.inference_mode():
            t0_cold = time.perf_counter_ns()
            _ = model(x1)
            cold_ms = (time.perf_counter_ns() - t0_cold) / 1e6

        steady_raw = measure_indexed_latencies(
            fn=lambda _i: model(x1),
            warmup_iters=steady_warmup,
            timed_iters=steady_iters,
            disable_gc=disable_gc,
        )
        steady_stats = summarize_latencies(steady_raw)

    ratio = (cold_ms / steady_stats.median_ms) if steady_stats.median_ms > 0 else 0.0
    return CnnColdStartRecord(
        model_name=model_name,
        checkpoint_load_ms=load_ms,
        first_inference_ms=cold_ms,
        steady_state_median_ms=steady_stats.median_ms,
        cold_to_steady_ratio=ratio,
    )


def profile_cnn_layers(
    model: nn.Module,
    model_name: str,
    inputs: torch.Tensor,
    threads: int = 1,
    batch_size: int = 1,
    warmup_iters: int = 20,
    timed_iters: int = 100,
    torch_prof_warmup: int = 3,
    torch_prof_active: int = 10,
    hook_overhead_warn_pct: float = 15.0,
    disable_gc: bool = True,
    bootstrap_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    chrome_trace_path: Path | None = None,
    cold_start: CnnColdStartRecord | None = None,
) -> CnnLayerProfileReport:
    """Profile leaf module latencies, activations, FLOP intensity, and ATen ops on CPU."""
    if inputs.ndim != 4 or inputs.shape[0] < 1:
        raise ValueError(f"Expected 4D tensor [N>=1, 1, H, W], got {tuple(inputs.shape)}")

    image_size = int(inputs.shape[-1])
    model_cpu = model.to("cpu").eval()
    total_params, trainable_params, size_mb = model_stats(model_cpu)
    total_flops = estimate_flops(model_cpu, input_size=image_size)

    # Pre-slice real CPU batches outside timing
    host_x = inputs.detach().to(device="cpu", dtype=torch.float32).contiguous()
    n_avail = int(host_x.shape[0])
    batches: list[torch.Tensor] = []
    if batch_size == 1:
        for i in range(n_avail):
            batches.append(host_x[i : i + 1].contiguous())
    else:
        n_batches = max(1, (n_avail + batch_size - 1) // batch_size)
        for b in range(n_batches):
            idx = (torch.arange(b * batch_size, (b + 1) * batch_size) % n_avail).to(torch.long)
            batches.append(host_x.index_select(0, idx).contiguous())
    n_batches = len(batches)

    with thread_limits(threads):
        # 1. Hookless clean measurement (official total timing reference)
        with torch.inference_mode():
            hookless_ms = measure_indexed_latencies(
                fn=lambda idx: model_cpu(batches[idx % n_batches]),
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                disable_gc=disable_gc,
            )
        hookless_stats = summarize_latencies(
            hookless_ms,
            per_sample_divisor=batch_size,
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            seed=seed,
        )

        # 2. Leaf module forward hooks
        leaf_entries: list[tuple[str, nn.Module]] = [
            (name or mod.__class__.__name__, mod)
            for name, mod in model_cpu.named_modules()
            if len(list(mod.children())) == 0
        ]
        mod_to_name = {mod: name for name, mod in leaf_entries}

        active_starts: dict[str, int] = {}
        iter_accum_ns: dict[str, int] = {name: 0 for name, _ in leaf_entries}
        iter_call_counts: dict[str, int] = {name: 0 for name, _ in leaf_entries}
        iter_flops: dict[str, int] = {name: 0 for name, _ in leaf_entries}
        in_shapes: dict[str, str] = {}
        out_shapes: dict[str, str] = {}
        act_bytes: dict[str, int] = {}
        last_calls: dict[str, int] = {name: 0 for name, _ in leaf_entries}
        last_flops: dict[str, int] = {name: 0 for name, _ in leaf_entries}

        per_iter_ms: dict[str, list[float]] = {name: [] for name, _ in leaf_entries}

        def _pre_hook(mod: nn.Module, inp: tuple[Any, ...]) -> None:
            l_name = mod_to_name[mod]
            if l_name not in in_shapes and inp:
                s_str, _ = _tensor_shape_and_bytes(inp[0])
                in_shapes[l_name] = s_str
            active_starts[l_name] = time.perf_counter_ns()

        def _post_hook(mod: nn.Module, inp: tuple[Any, ...], out: Any) -> None:
            t_end = time.perf_counter_ns()
            l_name = mod_to_name[mod]
            t_start = active_starts.pop(l_name, t_end)
            iter_accum_ns[l_name] += t_end - t_start
            iter_call_counts[l_name] += 1
            iter_flops[l_name] += _analytical_layer_flops(mod, out)
            s_out, b_out = _tensor_shape_and_bytes(out)
            out_shapes[l_name] = s_out
            act_bytes[l_name] = max(act_bytes.get(l_name, 0), b_out)

        handles = []
        for _, mod in leaf_entries:
            handles.append(mod.register_forward_pre_hook(_pre_hook))
            handles.append(mod.register_forward_hook(_post_hook))

        hooked_totals_ms: list[float] = []
        try:
            with disabled_gc(enabled=disable_gc):
                with torch.inference_mode():
                    for idx in range(warmup_iters):
                        for l_name in iter_accum_ns:
                            iter_accum_ns[l_name] = 0
                            iter_call_counts[l_name] = 0
                            iter_flops[l_name] = 0
                        _ = model_cpu(batches[idx % n_batches])

                    for idx in range(timed_iters):
                        for l_name in iter_accum_ns:
                            iter_accum_ns[l_name] = 0
                            iter_call_counts[l_name] = 0
                            iter_flops[l_name] = 0
                        t0_ns = time.perf_counter_ns()
                        _ = model_cpu(batches[idx % n_batches])
                        hooked_totals_ms.append((time.perf_counter_ns() - t0_ns) / 1e6)
                        for l_name in iter_accum_ns:
                            per_iter_ms[l_name].append(iter_accum_ns[l_name] / 1e6)
                            last_calls[l_name] = iter_call_counts[l_name]
                            last_flops[l_name] = iter_flops[l_name]
        finally:
            for h in handles:
                h.remove()

        hooked_stats = summarize_latencies(hooked_totals_ms, per_sample_divisor=batch_size, seed=seed)
        hook_overhead_pct = (
            ((hooked_stats.median_ms - hookless_stats.median_ms) / hookless_stats.median_ms) * 100.0
            if hookless_stats.median_ms > 0.0
            else 0.0
        )
        exceeds_warn = hook_overhead_pct > hook_overhead_warn_pct
        if exceeds_warn:
            logger.warning(
                "CNN '%s' leaf forward hook overhead is %.2f%% (> %.1f%% threshold); "
                "use torch.profiler ATen breakdown alongside hookless totals as timing ground truth.",
                model_name,
                hook_overhead_pct,
                hook_overhead_warn_pct,
            )

        # Aggregate layer statistics
        layer_summaries: list[tuple[str, nn.Module, LatencyStats]] = []
        sum_layers_ms = 0.0
        for l_name, mod in leaf_entries:
            l_st = summarize_latencies(
                per_iter_ms[l_name],
                per_sample_divisor=batch_size,
                bootstrap_resamples=min(200, bootstrap_resamples),
                confidence_level=confidence_level,
                seed=seed,
            )
            layer_summaries.append((l_name, mod, l_st))
            sum_layers_ms += l_st.median_ms

        layer_records: list[CnnLayerProfileRecord] = []
        category_ms: dict[str, float] = {}
        max_act_bytes = 0
        for l_name, mod, l_st in layer_summaries:
            cat = _classify_layer(mod)
            category_ms[cat] = category_ms.get(cat, 0.0) + l_st.median_ms
            l_params = sum(int(p.numel()) for p in mod.parameters(recurse=False))
            l_param_mb = sum(p.numel() * p.element_size() for p in mod.parameters(recurse=False)) / 1e6
            a_bytes = act_bytes.get(l_name, 0)
            max_act_bytes = max(max_act_bytes, a_bytes)
            l_flops = int(last_flops.get(l_name, 0) // max(1, batch_size))
            share_pct = (l_st.median_ms / sum_layers_ms * 100.0) if sum_layers_ms > 0 else 0.0
            ms_per_mf = (l_st.median_ms / (l_flops / 1e6)) if l_flops > 0 else None

            layer_records.append(
                CnnLayerProfileRecord(
                    model_name=model_name,
                    layer_name=l_name,
                    layer_type=mod.__class__.__name__,
                    category=cat,
                    call_count=last_calls.get(l_name, 0),
                    input_shape=in_shapes.get(l_name, "unused"),
                    output_shape=out_shapes.get(l_name, "unused"),
                    params=l_params,
                    param_mb_fp32=l_param_mb,
                    activation_bytes=a_bytes,
                    activation_kb=float(a_bytes) / 1024.0,
                    flops=l_flops,
                    median_ms=l_st.median_ms,
                    q25_ms=l_st.q25_ms,
                    q75_ms=l_st.q75_ms,
                    iqr_ms=l_st.iqr_ms,
                    p95_ms=l_st.p95_ms,
                    share_of_layers_pct=share_pct,
                    ms_per_mflop=ms_per_mf,
                )
            )

        category_shares_pct = {
            cat: (ms_val / sum_layers_ms * 100.0) if sum_layers_ms > 0.0 else 0.0
            for cat, ms_val in category_ms.items()
        }

        # 3. torch.profiler C++/ATen breakdown (warmed first outside trace window)
        with torch.inference_mode():
            for idx in range(torch_prof_warmup):
                _ = model_cpu(batches[idx % n_batches])

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            with torch.inference_mode():
                for idx in range(torch_prof_active):
                    _ = model_cpu(batches[idx % n_batches])
                    prof.step()

        if chrome_trace_path is not None:
            chrome_trace_path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(chrome_trace_path))

        key_avgs = prof.key_averages()
        aten_table_text = key_avgs.table(sort_by="cpu_time_total", row_limit=40)

        # Compute percentages over self CPU time (self CPU avoids double-counting parent->child operators)
        total_self_cpu_us = sum(float(getattr(evt, "self_cpu_time_total", 0.0)) for evt in key_avgs)
        total_cpu_us = sum(float(getattr(evt, "cpu_time_total", 0.0)) for evt in key_avgs)

        aten_records: list[AtenOpProfileRecord] = []
        sorted_evts = sorted(
            key_avgs,
            key=lambda e: float(getattr(e, "self_cpu_time_total", 0.0)),
            reverse=True,
        )
        for evt in sorted_evts:
            cpu_us = float(getattr(evt, "cpu_time_total", 0.0))
            self_us = float(getattr(evt, "self_cpu_time_total", 0.0))
            mem_b = float(getattr(evt, "cpu_memory_usage", 0.0))
            self_mem_b = float(getattr(evt, "self_cpu_memory_usage", 0.0))
            evt_flops = int(getattr(evt, "flops", 0) or 0)
            aten_records.append(
                AtenOpProfileRecord(
                    model_name=model_name,
                    op_name=str(evt.key),
                    call_count=int(getattr(evt, "count", 0)),
                    cpu_time_total_ms=cpu_us / 1000.0,
                    self_cpu_time_ms=self_us / 1000.0,
                    share_cpu_time_pct=(cpu_us / total_cpu_us * 100.0) if total_cpu_us > 0 else 0.0,
                    share_self_cpu_time_pct=(self_us / total_self_cpu_us * 100.0) if total_self_cpu_us > 0 else 0.0,
                    cpu_memory_kb=mem_b / 1024.0,
                    self_cpu_memory_kb=self_mem_b / 1024.0,
                    flops=evt_flops,
                )
            )

    return CnnLayerProfileReport(
        model_name=model_name,
        threads=threads,
        batch_size=batch_size,
        image_size=image_size,
        total_params=total_params,
        trainable_params=trainable_params,
        model_size_mb=size_mb,
        max_activation_kb=float(max_act_bytes) / 1024.0,
        total_flops=total_flops,
        hookless_stats=hookless_stats,
        hooked_total_median_ms=hooked_stats.median_ms,
        sum_layers_median_ms=sum_layers_ms,
        hook_overhead_pct=hook_overhead_pct,
        hook_overhead_exceeds_warn=exceeds_warn,
        category_shares_pct=category_shares_pct,
        layers=tuple(layer_records),
        aten_ops=tuple(aten_records),
        aten_table_text=aten_table_text,
        chrome_trace_path=chrome_trace_path,
        cold_start=cold_start,
        raw_hookless_latencies_ms=tuple(float(v / batch_size) for v in hookless_ms),
    )


def profile_cnn_thread_scaling(
    model: nn.Module,
    model_name: str,
    inputs: torch.Tensor,
    threads_list: Sequence[int],
    batch_size: int = 1,
    warmup_iters: int = 20,
    timed_iters: int = 200,
    disable_gc: bool = True,
    bootstrap_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[CnnThreadScalingRecord, ...]:
    """Measure CPU CNN forward latency across thread counts relative to single-thread baseline."""
    model_cpu = model.to("cpu").eval()
    host_x = inputs.detach().to(device="cpu", dtype=torch.float32).contiguous()
    n_avail = int(host_x.shape[0])
    batches: list[torch.Tensor] = []
    if batch_size == 1:
        for i in range(n_avail):
            batches.append(host_x[i : i + 1].contiguous())
    else:
        n_batches = max(1, (n_avail + batch_size - 1) // batch_size)
        for b in range(n_batches):
            idx = (torch.arange(b * batch_size, (b + 1) * batch_size) % n_avail).to(torch.long)
            batches.append(host_x.index_select(0, idx).contiguous())
    n_batches = len(batches)

    raw_stats: list[tuple[int, LatencyStats]] = []
    for th in threads_list:
        with thread_limits(int(th)):
            with torch.inference_mode():
                lat_ms = measure_indexed_latencies(
                    fn=lambda idx: model_cpu(batches[idx % n_batches]),
                    warmup_iters=warmup_iters,
                    timed_iters=timed_iters,
                    disable_gc=disable_gc,
                )
        st = summarize_latencies(
            lat_ms,
            per_sample_divisor=batch_size,
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            seed=seed,
        )
        raw_stats.append((int(th), st))

    # Baseline for speedup is 1-thread run if present, else lowest thread count
    baseline_ms = next((st.median_ms for th, st in raw_stats if th == 1), raw_stats[0][1].median_ms)
    records: list[CnnThreadScalingRecord] = []
    for th, st in raw_stats:
        speedup = (baseline_ms / st.median_ms) if st.median_ms > 0.0 else 0.0
        eff = (speedup / float(th) * 100.0) if th > 0 else 0.0
        records.append(
            CnnThreadScalingRecord(
                model_name=model_name,
                threads=th,
                batch_size=batch_size,
                median_ms=st.median_ms,
                iqr_ms=st.iqr_ms,
                p95_ms=st.p95_ms,
                ci95_low_ms=st.ci95_low_ms,
                ci95_high_ms=st.ci95_high_ms,
                throughput_samples_s=st.throughput_samples_s,
                speedup_vs_1thread=speedup,
                efficiency_pct=eff,
            )
        )
    return tuple(records)
