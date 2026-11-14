"""Static tensor arena liveness analysis and MCU flash/SRAM/PSRAM budget reporting."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .cnn_graph import CnnIrGraph
from .config import CImpConfig


@dataclass(frozen=True, slots=True)
class TensorAllocation:
    """Static arena offset and byte size for one IR tensor."""

    tensor_name: str
    first_def: int
    last_use: int
    numel: int
    offset_f32: int
    bytes_f32: int
    offset_q15: int
    bytes_q15: int


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    """Complete static arena layout and embedded memory fit classification."""

    model_name: str
    weight_format: str
    weight_flash_bytes: int
    peak_arena_f32_bytes: int
    peak_arena_q15_bytes: int
    allocations: dict[str, TensorAllocation]
    fits_internal_sram_q15: bool
    fits_psram_q15: bool
    fits_flash: bool


def _align_up(offset: int, align: int = 16) -> int:
    """Round byte offset up to `align`-byte boundary."""
    return ((offset + align - 1) // align) * align


def _pack_intervals(
    intervals: list[tuple[str, int, int, int]],
    align: int = 16,
) -> tuple[dict[str, int], int]:
    """Greedy first-fit interval coloring assigning non-overlapping arena byte offsets."""
    # Sort by start step then descending size for tighter packing
    ordered = sorted(intervals, key=lambda item: (item[1], -item[3], item[0]))
    placed: list[tuple[int, int, int, int]] = []  # (first_def, last_use, offset, end_byte)
    offsets: dict[str, int] = {}
    peak = 0

    for name, start, end, size in ordered:
        aligned_size = _align_up(size, align)
        # Collect occupied intervals overlapping [start, end]
        overlapping = [
            (off, stop)
            for (s0, e0, off, stop) in placed
            if not (e0 < start or s0 > end)
        ]
        overlapping.sort()
        candidate = 0
        for occ_start, occ_end in overlapping:
            if candidate + aligned_size <= occ_start:
                break
            if candidate < occ_end:
                candidate = _align_up(occ_end, align)
        placed.append((start, end, candidate, candidate + aligned_size))
        offsets[name] = candidate
        peak = max(peak, candidate + aligned_size)

    return offsets, peak


def build_memory_plan(
    graph: CnnIrGraph,
    weight_format: str,
    weight_flash_bytes: int,
    cfg: CImpConfig,
) -> MemoryPlan:
    """Compute non-overlapping f32/q15 arena offsets via tensor liveness across `graph`."""
    shapes: dict[str, tuple[int, int, int]] = {graph.input_name: graph.input_shape}
    first_def: dict[str, int] = {graph.input_name: 0}
    last_use: dict[str, int] = {graph.input_name: 0}

    for idx, op in enumerate(graph.ops, start=1):
        for inp in op.inputs:
            last_use[inp] = max(last_use.get(inp, 0), idx)
        shapes[op.output] = op.out_shape
        first_def[op.output] = idx
        last_use[op.output] = idx

    # Output tensor stays live until the final step
    last_use[graph.output_name] = max(last_use[graph.output_name], len(graph.ops))

    f32_intervals: list[tuple[str, int, int, int]] = []
    q15_intervals: list[tuple[str, int, int, int]] = []

    for name, chw in shapes.items():
        numel = int(math.prod(chw))
        b_f32 = numel * 4
        # Final logits in q15 use int32 (4 bytes); all other tensors use int16 (2 bytes)
        elem_bytes_q = 4 if name == graph.output_name else 2
        b_q15 = numel * elem_bytes_q
        s0 = first_def[name]
        e0 = last_use[name]
        f32_intervals.append((name, s0, e0, b_f32))
        q15_intervals.append((name, s0, e0, b_q15))

    off_f32, peak_f32 = _pack_intervals(f32_intervals, align=16)
    off_q15, peak_q15 = _pack_intervals(q15_intervals, align=16)

    allocations: dict[str, TensorAllocation] = {}
    for name, s0, e0, b_f32 in f32_intervals:
        numel = b_f32 // 4
        b_q15 = numel * (4 if name == graph.output_name else 2)
        allocations[name] = TensorAllocation(
            tensor_name=name,
            first_def=s0,
            last_use=e0,
            numel=numel,
            offset_f32=off_f32[name],
            bytes_f32=b_f32,
            offset_q15=off_q15[name],
            bytes_q15=b_q15,
        )

    return MemoryPlan(
        model_name=graph.model_name,
        weight_format=weight_format,
        weight_flash_bytes=weight_flash_bytes,
        peak_arena_f32_bytes=peak_f32,
        peak_arena_q15_bytes=peak_q15,
        allocations=allocations,
        fits_internal_sram_q15=peak_q15 <= cfg.cnn.sram_budget_bytes,
        fits_psram_q15=peak_q15 <= cfg.cnn.psram_budget_bytes,
        fits_flash=weight_flash_bytes <= cfg.cnn.flash_budget_bytes,
    )
