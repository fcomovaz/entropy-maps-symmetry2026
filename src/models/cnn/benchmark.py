"""Standalone inference latency, throughput, memory, and FLOP benchmarking protocol."""

from __future__ import annotations

from dataclasses import dataclass
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


def benchmark_model_latency(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
    warmup_iters: int = 10,
    timed_iters: int = 50,
    threads: int = 1,
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

    Returns:
        Tuple of (list_of_latencies_ms, peak_vram_mb_or_none).
    """
    if device.type == "cpu":
        torch.set_num_threads(threads)

    model = model.to(device).eval()
    dummy_input = torch.randn(batch_size, 1, image_size, image_size, device=device)

    # Warmup
    with torch.inference_mode():
        for _ in range(warmup_iters):
            model(dummy_input)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    samples_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(timed_iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)

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
) -> BenchmarkRecord:
    """Run full benchmarking suite for a single model, batch size, and device.

    Args:
        model: Architecture instance.
        model_name: Name of the architecture.
        cfg: Validated CnnConfig instance.
        device: Target device.
        batch_size: Batch size for latency test.

    Returns:
        Populated BenchmarkRecord.
    """
    params, trainable, size_mb = model_stats(model)
    flops = estimate_flops(model, cfg.data.image_size)

    samples_ms, peak_vram = benchmark_model_latency(
        model=model,
        device=device,
        batch_size=batch_size,
        image_size=cfg.data.image_size,
        warmup_iters=cfg.benchmark.warmup_iters,
        timed_iters=cfg.benchmark.timed_iters,
        threads=cfg.benchmark.threads,
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
