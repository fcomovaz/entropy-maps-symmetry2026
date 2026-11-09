"""Standalone inference latency, throughput, memory, and FLOP benchmarking protocol."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from importlib.util import find_spec
import logging
import statistics
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .architectures.common import model_stats
from .config import CnnConfig

logger = logging.getLogger("research")


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """Immutable latency, throughput, and hardware resource profile."""

    model_name: str
    device: str
    batch_size: int
    image_size: int
    parameters: int
    trainable_parameters: int
    size_mb: float
    flops: int
    median_ms: float
    iqr_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput_samples_per_s: float
    peak_vram_mb: float | None


def estimate_flops(model: nn.Module, input_size: int = 128) -> int:
    """Estimate total forward FLOPs for an input of shape [1, 1, input_size, input_size].

    Uses PyTorch 2.x FlopCounterMode if available, or torchinfo, or standard layer calculation.
    """
    dummy_input = torch.zeros(1, 1, input_size, input_size)

    # 1. Try PyTorch native FlopCounterMode (torch >= 2.1)
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as mode:
            with torch.inference_mode():
                model(dummy_input)
        total_flops = mode.get_total_flops()
        if total_flops > 0:
            return int(total_flops)
    except Exception:
        pass

    # 2. Try torchinfo if installed
    if find_spec("torchinfo") is not None:
        try:
            from torchinfo import summary

            info = summary(model, input_size=(1, 1, input_size, input_size), verbose=0)
            if hasattr(info, "total_mult_adds") and info.total_mult_adds is not None:
                # 1 MAC = 2 FLOPs
                return int(info.total_mult_adds * 2)
        except Exception:
            pass

    # 3. Fallback: analytical estimation through hooks
    flops = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inp: Any, out: Any) -> None:
        nonlocal flops
        output_elements = out.numel()
        kernel_elements = module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]
        # Multiplications and additions: 2 * Cin * K * K * Hout * Wout
        flops += 2 * (output_elements // out.shape[1]) * kernel_elements * out.shape[1]

    def linear_hook(module: nn.Linear, inp: Any, out: Any) -> None:
        nonlocal flops
        flops += 2 * module.in_features * module.out_features

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(linear_hook))

    model.eval()
    with torch.inference_mode():
        model(dummy_input)

    for h in hooks:
        h.remove()

    return flops


def _prepare_input_batches(
    inputs: torch.Tensor | None,
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Prepare contiguous device-resident input batches of shape [batch_size, 1, H, W].

    When `inputs` is supplied, validates shape `[N, 1, H, W]` (`N >= 1`) and cycles across real
    samples outside the timed region. Falls back to synthetic standard-normal noise when `None`.
    """
    if inputs is None:
        return [torch.randn(batch_size, 1, image_size, image_size, device=device)]

    if inputs.ndim != 4 or inputs.shape[0] < 1 or tuple(inputs.shape[1:]) != (1, image_size, image_size):
        raise ValueError(
            f"Expected inputs shape [N>=1, 1, {image_size}, {image_size}], got {tuple(inputs.shape)}"
        )

    n_samples = int(inputs.shape[0])
    host_cpu = inputs.detach().to(dtype=torch.float32)
    batches: list[torch.Tensor] = []

    if batch_size == 1:
        for idx in range(n_samples):
            batches.append(host_cpu[idx : idx + 1].contiguous().to(device))
        return batches

    n_full_batches = (n_samples + batch_size - 1) // batch_size
    for b_idx in range(n_full_batches):
        start = (b_idx * batch_size) % n_samples
        indices = torch.arange(start, start + batch_size, dtype=torch.long) % n_samples
        batches.append(host_cpu.index_select(0, indices).contiguous().to(device))
    return batches


def benchmark_model_latency(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
    warmup_iters: int = 10,
    timed_iters: int = 50,
    threads: int = 1,
    inputs: torch.Tensor | None = None,
    disable_gc: bool = False,
) -> tuple[list[float], float | None]:
    """Measure raw forward latency distributions under strict isolation.

    Args:
        model: Evaluated nn.Module.
        device: Evaluation torch.device.
        batch_size: Batch size.
        image_size: Image height and width.
        warmup_iters: Warmup forwards prior to timing.
        timed_iters: Timed forward iterations.
        threads: Number of CPU threads (when running on CPU).
        inputs: Optional real input tensor of shape `[N, 1, image_size, image_size]`.
            When omitted, synthetic standard normal input is used for compatibility.
        disable_gc: When True, temporarily disables Python GC during measurement.

    Returns:
        Tuple of (list_of_latencies_ms, peak_vram_mb_or_none).
    """
    if device.type == "cpu":
        torch.set_num_threads(threads)

    model = model.to(device).eval()
    batches = _prepare_input_batches(
        inputs=inputs,
        batch_size=batch_size,
        image_size=image_size,
        device=device,
    )
    n_batches = len(batches)

    gc_was_enabled = gc.isenabled()
    if disable_gc and gc_was_enabled:
        gc.disable()

    samples_ms: list[float] = []
    try:
        # Warmup
        with torch.inference_mode():
            for i in range(warmup_iters):
                _ = model(batches[i % n_batches])

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        with torch.inference_mode():
            for i in range(timed_iters):
                batch = batches[i % n_batches]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0_ns = time.perf_counter_ns()
                _ = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                samples_ms.append((time.perf_counter_ns() - t0_ns) / 1e6)
    finally:
        if disable_gc and gc_was_enabled:
            gc.enable()

    peak_vram_mb: float | None = None
    if device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_mb = peak_bytes / (1024.0 * 1024.0)

    return samples_ms, peak_vram_mb


def run_benchmark(
    model: nn.Module,
    model_name: str,
    cfg: CnnConfig,
    device: torch.device,
    batch_size: int = 1,
    inputs: torch.Tensor | None = None,
    threads: int | None = None,
    warmup_iters: int | None = None,
    timed_iters: int | None = None,
    disable_gc: bool = False,
) -> BenchmarkRecord:
    """Run full benchmarking suite for a single model, batch size, and device.

    Args:
        model: Architecture instance.
        model_name: Name of the architecture.
        cfg: Validated CnnConfig instance.
        device: Target device.
        batch_size: Batch size for latency test.
        inputs: Optional real tensor `[N, 1, H, W]` instead of synthetic `randn`.
        threads: Optional CPU thread override (defaults to `cfg.benchmark.threads`).
        warmup_iters: Optional warmup override (defaults to `cfg.benchmark.warmup_iters`).
        timed_iters: Optional timed iterations override (defaults to `cfg.benchmark.timed_iters`).
        disable_gc: Whether to disable Python GC during timing.

    Returns:
        Populated BenchmarkRecord.
    """
    params, trainable, size_mb = model_stats(model)
    flops = estimate_flops(model, cfg.data.image_size)

    eff_threads = cfg.benchmark.threads if threads is None else int(threads)
    eff_warmup = cfg.benchmark.warmup_iters if warmup_iters is None else int(warmup_iters)
    eff_timed = cfg.benchmark.timed_iters if timed_iters is None else int(timed_iters)

    samples_ms, peak_vram = benchmark_model_latency(
        model=model,
        device=device,
        batch_size=batch_size,
        image_size=cfg.data.image_size,
        warmup_iters=eff_warmup,
        timed_iters=eff_timed,
        threads=eff_threads,
        inputs=inputs,
        disable_gc=disable_gc,
    )

    med_ms = float(statistics.median(samples_ms))
    mean_ms = float(statistics.mean(samples_ms))
    std_ms = float(statistics.stdev(samples_ms)) if len(samples_ms) > 1 else 0.0
    min_ms = float(min(samples_ms))
    max_ms = float(max(samples_ms))

    q75, q25 = np.percentile(samples_ms, [75, 25])
    iqr_ms = float(q75 - q25)

    # Throughput: samples per second based on mean batch time
    throughput = (batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0

    return BenchmarkRecord(
        model_name=model_name,
        device=str(device),
        batch_size=batch_size,
        image_size=cfg.data.image_size,
        parameters=params,
        trainable_parameters=trainable,
        size_mb=size_mb,
        flops=flops,
        median_ms=med_ms,
        iqr_ms=iqr_ms,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        throughput_samples_per_s=throughput,
        peak_vram_mb=peak_vram,
    )
